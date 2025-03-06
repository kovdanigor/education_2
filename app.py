# DOnt work

import networkx as nx
from ipysigma import Sigma
from shinyswatch import theme
from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_widget, render_plotly
import pandas as pd
import netfunction
import plotly.express as px
from faicons import icon_svg
import plotly.graph_objects as go

from qdrant_client import models, QdrantClient
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from rag_chat import create_vector_store, template, create_retrievers, format_docs

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Настройки страницы
ui.page_opts(
    title=ui.div(
        icon_svg("vector-square"),      # Иконка сети из faicons
        " Network Dashboard",
        style="display: flex; align-items: center;"
    ),
    fillable=True,
    id="page",
    theme=theme.journal
)

# Sidebar: Обработка данных и универсальные фильтры для графов
with ui.sidebar(width=400):
    # Обработка данных
    ui.HTML("<h5> ⚙️Обработка данных</h5>")
    ui.hr()

    with ui.card(full_screen=False):
        ui.input_file("file", "Загрузить данные:", accept=".xlsx", width='400px',
                      button_label='Обзор', placeholder='Файл отсутствует')

    ui.hr()

    # Универсальные фильтры для сетей
    with ui.card(full_screen=False):
        with ui.accordion(id="acc", multiple=True, open=False):
            with ui.accordion_panel('Фильтрация данных'):
                ui.input_date_range(
                    "pub_date", "Дата публикации вакансии:",
                    start="2024-01-01", end="2024-12-31",
                    min="2024-01-01", max="2024-12-31", width='400px'
                )
                ui.input_selectize("experience", "Опыт работы:",
                                   choices=[], multiple=True, width='400px')
                ui.input_selectize("region", "Регион:", choices=[],
                                   multiple=True, width=200)
                ui.input_selectize("employer", "Работодатель:",
                                   choices=[], multiple=True, width='400px')
                ui.input_selectize("specialty", "Название специальности:",
                                   choices=[], multiple=True, width='400px')
                ui.input_slider("salary", "Заработная плата:", min=0,
                                max=100000, value=[0, 100000])

    with ui.card(full_screen=False):
        with ui.accordion(id="acc2", multiple=True, open=False):
            with ui.accordion_panel('Фильтрация двумодального графа'):
                col_choices = ['Название специальности', 'Работодатель',
                               'Название региона', 'Федеральный округ']
                row_choices = ['Название специальности', 'Обработанные навыки',
                               'Работодатель', 'Название региона', 'Федеральный округ']
                ui.input_selectize(
                    "bipartite_col", "Выбор колонки:", choices=col_choices,
                    selected='Название специальности', width='400px')
                ui.input_selectize(
                    "bipartite_row", "Выбор строки:", choices=row_choices,
                    selected='Обработанные навыки', width='400px')

    with ui.card(full_screen=False):
        with ui.accordion(id="acc3", multiple=True, open=False):
            with ui.accordion_panel('Создание векторной базы'):
                ui.input_text("collection_name", "Имя коллекции для Qdrant:",
                              value="defaut_collection", placeholder="Введите имя коллекции")
                ui.input_text("qdrant_url", "URL кластера:",
                              value="https://ad51d191-06a9-407d-92e6-6183848591d5.us-east4-0.gcp.cloud.qdrant.io:6333",
                              placeholder="Введите URL Qdrant кластера")
                ui.input_password("qdrant_api_key", "API-токен кластера:",
                                  placeholder="Введите токен")
                ui.input_slider("filter_data", "Выборка для векторизации:", min=100,
                                max=2000, value=800, step=100)

                with ui.layout_columns(col_widths={"sm": (6, 6)}):
                    ui.input_action_button(
                        "create_collection", "Создать коллекцию", class_="btn btn-success btn-sm")
                    ui.input_action_button(
                        "delete_collection", "Удалить коллекцию", class_="btn btn-primary btn-sm")
    ui.hr()


# Реактивные вычисления и эффекты


@reactive.calc
def df():
    f = req(input.file())
    return pd.read_excel(f[0]['datapath'])


@reactive.calc
def processed_data():
    data = df()
    data['Обработанные навыки'] = data['Ключевые навыки'].apply(
        netfunction.parse_skills)
    data = data.dropna(subset=['Работодатель', 'Ключевые навыки'])
    data.reset_index(inplace=True, drop=True)
    data['Дата публикации'] = pd.to_datetime(data['Дата публикации'])
    data["Федеральный округ"] = data["Название региона"].apply(
        netfunction.get_federal_district)
    data['Данные'] = data.apply(lambda row:
                                f"{row['Работодатель']} ищет "
                                f"{row['Название специальности']} с {row['Ключевые навыки']}", axis=1)
    return data

# --- Остальное


@reactive.effect
def update_selects():
    # Получаем текущие выбранные значения из UI
    row_val = input.bipartite_row()
    col_val = input.bipartite_col()

    # Обновляем варианты для строки: исключаем выбранное значение из колонки
    new_row_choices = [r for r in row_choices if r != col_val]
    # Если текущий выбор строки недопустим, заменяем на первый допустимый вариант
    if row_val not in new_row_choices:
        row_val = new_row_choices[0] if new_row_choices else None

    # Обновляем варианты для колонки: исключаем выбранное значение из строки
    new_col_choices = [c for c in col_choices if c != row_val]
    # Если текущий выбор колонки недопустим, заменяем на первый допустимый вариант
    if col_val not in new_col_choices:
        col_val = new_col_choices[0] if new_col_choices else None

    # Обновляем UI компоненты с новыми вариантами и выбранными значениями
    ui.update_selectize(
        "bipartite_row", choices=new_row_choices, selected=row_val)
    ui.update_selectize(
        "bipartite_col", choices=new_col_choices, selected=col_val)


@reactive.effect
def update_filter_choices():
    data = processed_data()
    exp_choices = sorted(data["Опыт работы"].dropna().unique().tolist())
    region_choices = sorted(
        data["Название региона"].dropna().unique().tolist())
    employer_choices = sorted(data['Работодатель'].dropna().unique().tolist())
    specialty_choices = sorted(
        data["Название специальности"].dropna().unique().tolist())

    ui.update_selectize("experience", choices=exp_choices)
    ui.update_selectize("region", choices=region_choices)
    ui.update_selectize("employer", choices=employer_choices)
    ui.update_selectize("specialty", choices=specialty_choices)


@reactive.effect
def update_date_range():
    data = processed_data()
    if not data.empty:
        dates = data['Дата публикации']
        min_date = dates.min().date().isoformat()
        max_date = dates.max().date().isoformat()
        ui.update_date_range("pub_date", min=min_date,
                             max=max_date, start=min_date, end=max_date)


@reactive.effect
def update_salary_range():
    data = processed_data()
    if not data.empty:
        min_salary = int(data['Заработная плата'].min())
        max_salary = int(data['Заработная плата'].max())
        ui.update_slider("salary", min=min_salary,
                         max=max_salary, value=[min_salary, max_salary])


@reactive.calc
def filtered_data():
    data = processed_data()
    if input.pub_date():
        start_date, end_date = input.pub_date()
        data = data[(data['Дата публикации'] >= pd.to_datetime(start_date)) &
                    (data['Дата публикации'] <= pd.to_datetime(end_date))]
    if input.experience():
        data = data[data['Опыт работы'].isin(input.experience())]
    if input.region():
        data = data[data['Название региона'].isin(input.region())]
    if input.salary():
        min_salary, max_salary = input.salary()
        data = data[(data['Заработная плата'] >= min_salary) &
                    (data['Заработная плата'] <= max_salary)]
    if input.employer():
        data = data[data['Работодатель'].isin(input.employer())]
    if input.specialty():
        data = data[data['Название специальности'].isin(input.specialty())]
    return data


@reactive.calc
def semantic_cooccurrence_matrix():
    data = filtered_data()
    if data.empty:
        return pd.DataFrame()
    return netfunction.create_co_occurrence_matrix(data, 'Обработанные навыки')


@reactive.calc
def semantic_graph():
    matrix = semantic_cooccurrence_matrix()
    if matrix.empty:
        return None
    G = nx.from_pandas_adjacency(matrix)
    return G


@reactive.calc
def bipartite_matrix_custom():
    data = filtered_data()
    if data.empty:
        return pd.DataFrame()
    # Если селекты не выбраны, используем дефолтные значения
    col_var = input.bipartite_col() or 'Название специальности'
    row_var = input.bipartite_row() or 'Обработанные навыки'
    return netfunction.create_group_values_matrix(data, col_var, row_var)


@reactive.calc
def bipartite_graph():
    matrix = bipartite_matrix_custom()
    if matrix.empty:
        return None
    return netfunction.create_bipartite_graph(matrix)


# Реактивные функции для работы с Qdrant
@reactive.effect
@reactive.event(input.create_collection)
def create_qdrant_collection():
    try:
        qdrant_client = QdrantClient(
            url=input.qdrant_url(),
            api_key=input.qdrant_api_key(),
        )

        try:
            data = processed_data()
            data = data[['Название региона', 'Данные', 'Опыт работы']].sample(
                input.filter_data(), random_state=1)
            loader = DataFrameLoader(data, page_content_column="Данные")
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter()
            split_documents = splitter.split_documents(documents)
        except Exception as e:
            ui.notification_show(
                f"❌ Данные не загружены: {str(e)}", duration=10, type="error")
            return

        if not qdrant_client.collection_exists(collection_name=input.collection_name()):
            qdrant_client.create_collection(
                collection_name=input.collection_name(),
                vectors_config=models.VectorParams(
                    size=384, distance=models.Distance.COSINE)
            )
            create_vector_store(
                qdrant_client, input.collection_name(), embeddings, split_documents)
            ui.notification_show("✅ Коллекция успешно создана!", duration=5)
        else:
            ui.notification_show("ℹ️ Коллекция уже существует!", duration=5)

    except Exception as e:
        print(e)
        ui.notification_show(
            f"❌ Ошибка создания: {str(e)}", duration=10, type="error")


@reactive.effect
@reactive.event(input.delete_collection)
def delete_qdrant_collection():
    try:
        qdrant_client = QdrantClient(
            url=input.qdrant_url(),
            api_key=input.qdrant_api_key(),
        )

        if qdrant_client.collection_exists(collection_name=input.collection_name()):
            qdrant_client.delete_collection(
                collection_name=input.collection_name())
            ui.notification_show("🗑️ Коллекция удалена!", duration=5)
        else:
            ui.notification_show("ℹ️ Коллекция не найдена!", duration=5)

    except Exception as e:
        ui.notification_show(
            f"❌ Ошибка удаления: {str(e)}", duration=10, type="error")


# --- Панели ---

ui.nav_spacer()
with ui.nav_panel("Данные", icon=icon_svg("table")):
    with ui.card(full_screen=True):
        ui.card_header("📖 Загруженные данные")

        @render.data_frame
        def table():
            return render.DataTable(processed_data(), filters=True, height='550px')


with ui.nav_panel("Визуализация", icon=icon_svg("chart-bar")):
    with ui.layout_columns(col_widths=(12, 12)):
        with ui.card(full_screen=True):
            ui.card_header(
                "💰 Распределение средней зарплаты: Федеральный округ → Специальность → Опыт работы")

            @render_plotly
            def sankey_chart():
                data = filtered_data()
                if data.empty:
                    return px.scatter(title="Нет данных для отображения ❌")

                df_sankey = data.groupby(["Федеральный округ", "Название специальности", "Опыт работы"])[
                    "Заработная плата"].agg(netfunction.nonzero_mean).reset_index()

                unique_districts = list(
                    df_sankey["Федеральный округ"].unique())
                unique_specialties = list(
                    df_sankey["Название специальности"].unique())
                unique_experience = list(df_sankey["Опыт работы"].unique())

                nodes = unique_districts + unique_specialties + unique_experience
                node_indices = {name: i for i, name in enumerate(nodes)}

                source_districts = df_sankey["Федеральный округ"].map(
                    node_indices).tolist()
                target_specialties = df_sankey["Название специальности"].map(
                    node_indices).tolist()
                values_districts = df_sankey["Заработная плата"].tolist()

                source_specialties = df_sankey["Название специальности"].map(
                    node_indices).tolist()
                target_experience = df_sankey["Опыт работы"].map(
                    node_indices).tolist()
                values_specialties = df_sankey["Заработная плата"].tolist()

                source = source_districts + source_specialties
                target = target_specialties + target_experience
                value = values_districts + values_specialties

                palette = px.colors.qualitative.Set2
                node_colors = {node: palette[i % len(
                    palette)] for i, node in enumerate(nodes)}

                opacity = 0.4
                link_colors = [node_colors[nodes[src]].replace(
                    ")", f", {opacity})").replace("rgb", "rgba") for src in source]

                fig = go.Figure(go.Sankey(
                    valueformat=".0f",
                    node=dict(
                        pad=15,
                        thickness=25,
                        line=dict(color="black", width=0.7),
                        label=nodes,
                        color=[node_colors[node]
                               for node in nodes],
                        hoverlabel=dict(
                            font=dict(size=14, family="Arial", color="black", weight="bold")),
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=link_colors
                    )
                ))

                fig.update_layout(
                    title=None,
                    font=dict(size=14, family="Arial", color="black",
                              weight="bold"),
                    plot_bgcolor="white"
                )

                return fig

        with ui.card(full_screen=True):
            ui.card_header("📈 Динамика публикации вакансий по специальностям")

            @render_plotly
            def vacancies_trend():
                data = filtered_data()
                if data.empty:
                    return px.scatter(title="Нет данных для отображения")

                df_grouped = data.groupby(
                    [pd.Grouper(key="Дата публикации", freq="M"),
                     "Название специальности"]
                ).size().reset_index(name="Количество вакансий")

                fig = px.line(
                    df_grouped,
                    x="Дата публикации",
                    y="Количество вакансий",
                    color="Название специальности",
                    title="",
                    template="plotly_white",
                    markers=True
                ).update_layout(xaxis_title=None, yaxis_title=None, title=None)
                return fig


# Панель с графами с дополнительными фильтрами для настройки визуализации Sigma
with ui.nav_panel("Сеть", icon=icon_svg('circle-nodes')):
    with ui.navset_card_underline(id="selected_navset_card_underline1"):

        # Панель для двумодального графа
        with ui.nav_panel("Двумодальный граф"):
            with ui.layout_columns(col_widths=(3, 9)):
                # Левая колонка: фильтры для Sigma-визуализации двумодального графа
                with ui.card(full_screen=False):
                    ui.card_header("🔎 Фильтры визуализации")

                    ui.input_slider(
                        "edge_threshold_dm", "Порог силы связей:",
                        min=0, max=500, value=0, width=250
                    )

                    ui.input_selectize(
                        "node_size_dm", "Метрика размера узла:",
                        choices=["degree_centrality",
                                 "closeness_centrality", "betweenness_centrality"], width=250
                    )
                    ui.input_slider(
                        "node_size_range_dm", "Диапазон размера узла:",
                        min=1, max=50, value=[3, 15], width=250
                    )

                    ui.input_slider(
                        "edge_size_range_dm", "Диапазон размера ребра:",
                        min=1, max=50, value=[1, 10], width=250
                    )

                    ui.input_selectize(
                        "node_size_scale_dm", "Масштаб размера узла:",
                        choices=["lin", "log", "pow", "sqrt"], width=250
                    )
                    ui.input_slider(
                        "louvain_resolution_dm", "Разрешение Louvain:",
                        min=0, max=2, value=1, step=0.1, width=250
                    )
                # Правая колонка: визуализация графа
                with ui.card(full_screen=True):
                    ui.card_header("🔗 Граф")

                    @render_widget
                    def widget():
                        G = bipartite_graph()
                        if G is None:
                            ui.notification_show(
                                "❌ Нет данных для построения графа",
                                type="error", duration=10
                            )
                            return None
                        # Выбор метрики для размера узлов

                        threshold = input.edge_threshold_dm() or 0
                        G = netfunction.filter_graph(G, threshold)

                        metric_choice = input.node_size_dm()
                        if metric_choice == "degree_centrality":
                            metric = nx.degree_centrality(G)
                        elif metric_choice == "closeness_centrality":
                            metric = nx.closeness_centrality(G)
                        elif metric_choice == "betweenness_centrality":
                            metric = nx.betweenness_centrality(G)
                        else:
                            metric = nx.degree_centrality(G)
                        node_size_values = list(metric.values())

                        return Sigma(
                            G,
                            node_size=node_size_values,
                            node_size_range=input.node_size_range_dm() or (1, 10),
                            edge_size_range=input.edge_size_range_dm() or (1, 10),
                            node_size_scale=input.node_size_scale_dm() or "lin",
                            node_metrics={"community": {
                                "name": "louvain", "resolution": input.louvain_resolution_dm() or 1}},
                            node_color="community",
                            hide_edges_on_move=True,
                            edge_size='weight',
                            node_border_color_from='node',
                            hide_info_panel=True
                        )

        # Панель для одномодального (семантического) графа
        with ui.nav_panel("Одномодальный семантический граф"):
            with ui.layout_columns(col_widths=(3, 9)):
                # Левая колонка: фильтры для Sigma-визуализации одномодального графа
                with ui.card(full_screen=False):
                    ui.card_header("🔎 Фильтры визуализации")
                    ui.input_slider(
                        "edge_threshold_om", "Порог силы связей:",
                        min=0, max=500, value=0, width=250
                    )
                    ui.input_selectize(
                        "node_size_om", "Метрика размера узла:",
                        choices=["degree_centrality",
                                 "closeness_centrality", "betweenness_centrality"], width=250
                    )
                    ui.input_slider(
                        "node_size_range_om", "Диапазон размера узла:",
                        min=1, max=50, value=[3, 15], width=250
                    )

                    ui.input_slider(
                        "edge_size_range_om", "Диапазон размера ребра:",
                        min=1, max=50, value=[1, 10], width=250
                    )

                    ui.input_selectize(
                        "node_size_scale_om", "Масштаб размера узла:",
                        choices=["lin", "log", "pow", "sqrt"], width=250
                    )
                    ui.input_slider(
                        "louvain_resolution_om", "Разрешение Louvain:",
                        min=0, max=2, value=1, step=0.1, width=250
                    )
                # Правая колонка: визуализация семантического графа
                with ui.card(full_screen=True):
                    ui.card_header("🔗 Граф навыков")

                    @render_widget
                    def widget_semantic():
                        G = semantic_graph()

                        if G is None:
                            ui.notification_show("❌ Нет данных для построения графа",
                                                 type="error", duration=10
                                                 )
                            return None

                        threshold = input.edge_threshold_om() or 0
                        G = netfunction.filter_graph(G, threshold)
                        # Выбор метрики для размера узлов
                        metric_choice = input.node_size_om()
                        if metric_choice == "degree_centrality":
                            metric = nx.degree_centrality(G)
                        elif metric_choice == "closeness_centrality":
                            metric = nx.closeness_centrality(G)
                        elif metric_choice == "betweenness_centrality":
                            metric = nx.betweenness_centrality(G)
                        else:
                            metric = nx.degree_centrality(G)
                        node_size_values = list(metric.values())

                        return Sigma(
                            G,
                            node_size=node_size_values,
                            node_size_range=input.node_size_range_om() or (3, 15),
                            edge_size_range=input.edge_size_range_om() or (1, 10),
                            node_size_scale=input.node_size_scale_om() or "lin",
                            node_metrics={"community": {
                                "name": "louvain", "resolution": input.louvain_resolution_om() or 1}},
                            node_color="community",
                            hide_edges_on_move=True,
                            edge_size='weight',
                            node_border_color_from='node',
                            hide_info_panel=True
                        )

# --- Рекомендации ----

# Рекомендации рефакторинг
# Общая функция для создания графика визуализации


def create_bar_chart(G, node, node_type, top_n, recommendation_func, x_label, title_template):
    """
    Создает график-бар с визуализацией рекомендаций.

    :param G: Граф, в котором ищутся рекомендации.
    :param node: Выбранный узел.
    :param node_type: Тип узла ("Специальность" или "Навык").
    :param top_n: Количество наблюдений (верхних рекомендаций).
    :param recommendation_func: Функция для получения рекомендаций.
    :param x_label: Подпись для оси X.
    :param title_template: Шаблон заголовка графика (с параметрами {top_n} и {node}).
    :param error_message: Сообщение, если узел не выбран или произошла ошибка.
    :return: Объект графика Plotly.
    """
    if not node:
        return px.bar(x=["Нет выделенных узлов"], y=[0], template="plotly_white").update_layout()

    level_target = "first" if node_type == "Колонка" else "second"

    try:
        recs = recommendation_func(
            G, node, level_target=level_target, top_n=top_n)
        recs.sort(key=lambda x: x[1], reverse=False)
        nodes, similarities = zip(*recs)

        if x_label != 'Вес':
            similarities = [el * 100 for el in similarities]
    except:
        return px.bar(x=["Нет выделенных узлов"], y=[0], template="plotly_white").update_layout()

    unique_nodes = list(set(nodes))
    colors = px.colors.qualitative.G10
    color_map = {n: colors[i % len(colors)]
                 for i, n in enumerate(unique_nodes)}

    fig = px.bar(
        y=nodes,
        x=similarities,
        labels={'x': x_label, 'y': ''},
        title=title_template.format(top_n=top_n, node=node),
        color=nodes,
        template="plotly_white",
        color_discrete_map=color_map
    ).update_layout(
        showlegend=False,
        title_x=0.5,  # Центрирование заголовка
        title_font=dict(  # Настройка шрифта заголовка
            size=14,  # Размер шрифта (меньше стандартного)
            # family="Arial",  # Тип шрифта
            color="black",  # Цвет шрифта
            weight="bold"  # Жирный шрифт
        )
    )
    return fig


# --- Код интерфейса остаётся без изменений ---
with ui.nav_panel("Рекомендация", icon=icon_svg('diagram-project')):
    with ui.navset_card_underline(id="selected_navset_card_underline"):
        with ui.nav_panel("Рекомендация схожих узлов"):
            with ui.layout_columns(col_widths=(6, 6)):
                with ui.card(full_screen=True):
                    ui.card_header("📊 Рекомендация схожих узлов № 1")

                    with ui.layout_columns(col_widths={"sm": (12, 12)}):
                        ui.input_selectize(
                            "node_1", "Выбрать узел:", choices=[], width='750px')
                        ui.input_numeric(
                            "obs_1", "Количество наблюдений:", 3, min=1, max=30, width="750px")
                    ui.hr()

                    @reactive.effect
                    def update_node_choices_1():
                        matrix = bipartite_matrix_custom()
                        if matrix.empty:
                            ui.update_selectize("node_1", choices=[])
                        else:
                            choices = list(matrix.columns) + list(matrix.index)
                            ui.update_selectize("node_1", choices=choices)

                    @render_plotly
                    def recommendations_plot_1():
                        if filtered_data().empty:
                            ui.notification_show(
                                "❌ Нет данных, соответствующих выбранным фильтрам", type="error", duration=10)
                            return None

                        try:
                            G = bipartite_graph()
                            node = input.node_1()
                            top_n = input.obs_1()
                            bipartite_value = 'Колонка' if G.nodes[node]["bipartite"] == 1 else 'Строка'
                        except:
                            return None

                        return create_bar_chart(
                            G=G,
                            node=node,
                            node_type=bipartite_value,
                            top_n=top_n,
                            recommendation_func=netfunction.recommend_similar_nodes,
                            x_label='Сходство в %',
                            title_template='Топ {top_n} схожих узлов для "{node}"'
                        )

                with ui.card(full_screen=True):
                    ui.card_header("📊 Рекомендация схожих узлов № 2")

                    with ui.layout_columns(col_widths={"sm": (12, 12)}):
                        ui.input_selectize(
                            "node_2", "Выбрать узел:", choices=[], width='750px')
                        ui.input_numeric(
                            "obs_2", "Количество наблюдений:", 3, min=1, max=30, width='750px')
                    ui.hr()

                    @reactive.effect
                    def update_node_choices_2():
                        matrix = bipartite_matrix_custom()
                        if matrix.empty:
                            ui.update_selectize("node_2", choices=[])
                        else:
                            choices = list(matrix.columns) + list(matrix.index)
                            ui.update_selectize("node_2", choices=choices)

                    @render_plotly
                    def recommendations_plot_2():
                        if filtered_data().empty:
                            ui.notification_show("❌ Нет данных, соответствующих выбранным фильтрам",
                                                 type="error", duration=10)
                            return None

                        try:
                            G = bipartite_graph()
                            node = input.node_2()
                            top_n = input.obs_2()
                            bipartite_value = 'Колонка' if G.nodes[node]["bipartite"] == 1 else 'Строка'
                        except:
                            return None

                        return create_bar_chart(
                            G=G,
                            node=node,
                            node_type=bipartite_value,
                            top_n=top_n,
                            recommendation_func=netfunction.recommend_similar_nodes,
                            x_label='Сходство в %',
                            title_template='Топ {top_n} схожих узлов для "{node}"'
                        )

        with ui.nav_panel("Рекомендация соседних узлов"):
            with ui.layout_columns(col_widths=(6, 6)):
                with ui.card(full_screen=True):
                    ui.card_header("📊 Рекомендация соседних узлов № 1")

                    with ui.layout_columns(col_widths={"sm": (12, 12)}):
                        ui.input_selectize(
                            "node_3", "Выбрать узел:", choices=[], width='750px')
                        ui.input_numeric(
                            "obs_3", "Количество наблюдений:", 3, min=1, max=30, width='750px')
                    ui.hr()

                    @reactive.effect
                    def update_node_choices_3():
                        matrix = bipartite_matrix_custom()
                        if matrix.empty:
                            ui.update_selectize("node_3", choices=[])
                        else:
                            choices = list(matrix.columns) + list(matrix.index)
                            ui.update_selectize("node_3", choices=choices)

                    @render_plotly
                    def neighbor_recommendations_plot_1():
                        if filtered_data().empty:
                            ui.notification_show("❌ Нет данных, соответствующих выбранным фильтрам",
                                                 type="error",
                                                 duration=10)
                            return None

                        try:
                            G = bipartite_graph()
                            node = input.node_3()
                            top_n = input.obs_3()
                            bipartite_value = 'Колонка' if G.nodes[node]["bipartite"] == 1 else 'Строка'
                        except:
                            return None

                        return create_bar_chart(
                            G=G,
                            node=node,
                            node_type=bipartite_value,
                            top_n=top_n,
                            recommendation_func=netfunction.neighbor_recommendations,
                            x_label='Вес',
                            title_template='Топ {top_n} соседних узла "{node}"'
                        )

                # Новый блок для второй рекомендации соседних узлов
                with ui.card(full_screen=True):
                    ui.card_header("📊 Рекомендация соседних узлов № 2")

                    with ui.layout_columns(col_widths={"sm": (12, 12)}):
                        ui.input_selectize(
                            "node_4", "Выбрать узел:", choices=[], width='750px')
                        ui.input_numeric(
                            "obs_4", "Количество наблюдений:", 3, min=1, max=30, width='750px')
                    ui.hr()

                    @reactive.effect
                    def update_node_choices_4():
                        matrix = bipartite_matrix_custom()
                        if matrix.empty:
                            ui.update_selectize("node_4", choices=[])
                        else:
                            choices = list(matrix.columns) + list(matrix.index)
                            ui.update_selectize("node_4", choices=choices)

                    @render_plotly
                    def neighbor_recommendations_plot_2():
                        if filtered_data().empty:
                            ui.notification_show("❌ Нет данных, соответствующих выбранным фильтрам",
                                                 type="error",
                                                 duration=10)
                            return None

                        try:
                            G = bipartite_graph()
                            node = input.node_4()
                            top_n = input.obs_4()
                            bipartite_value = 'Колонка' if G.nodes[node]["bipartite"] == 1 else 'Строка'
                        except:
                            return None

                        return create_bar_chart(
                            G=G,
                            node=node,
                            node_type=bipartite_value,
                            top_n=top_n,
                            recommendation_func=netfunction.neighbor_recommendations,
                            x_label='Вес',
                            title_template='Топ {top_n} соседних узла для "{node}"'
                        )


prompt = ChatPromptTemplate.from_template(template)


@reactive.effect
def update_models():
    if input.base_url1() == "https://bothub.chat/api/v2/openai/v1":
        models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini",
                  "o3-mini-high", 'o1-mini',
                  "claude-3.7-sonnet:thinking",
                  "claude-3.5-haiku",
                  "deepseek-chat",
                  "deepseek-r1",
                  "qwen-2.5-72b-instruct"]
        ui.update_selectize("chat_model", choices=models)
    elif input.base_url1() == "https://openrouter.ai/api/v1":
        models = ["cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
                  "deepseek/deepseek-chat:free",
                  "deepseek/deepseek-r1:free",
                  "google/gemini-2.0-flash-thinking-exp:free",
                  'nousresearch/deephermes-3-llama-3-8b-preview:free',
                  'qwen/qwq-32b:free']
        ui.update_selectize("chat_model", choices=models)


with ui.nav_panel("Чат-бот", icon=icon_svg('robot')):
    with ui.layout_columns(col_widths=(4, 8)):
        # Левая колонка: Фильтры для чат-бота
        with ui.card(full_screen=False):
            ui.card_header("🔎 Фильтры для чат-бота")
            ui.input_password("chat_token", "API-токен сервиса:",
                              width='400px', placeholder="Введите токен")
            ui.input_selectize("chat_model", "Языковая модель:",
                               choices=[], width='400px')
            ui.input_selectize("base_url1", "Базовый URL-адрес сервиса:",
                               choices=["https://bothub.chat/api/v2/openai/v1",
                                        "https://openrouter.ai/api/v1"],
                               selected='https://openrouter.ai/api/v1', width='400px')
            ui.input_slider("temp", "Температура:", min=0,
                            max=1, value=0, step=0.1, width='400px')

        # Правая колонка: Чат-бот
        with ui.card(full_screen=True):
            ui.card_header("🤖 Чат-бот")
            welcome = ui.markdown("Hi!")
            chat = ui.Chat(id="chat", messages=[welcome])
            chat.ui(placeholder='Введите запрос...',
                    width='min(850px, 100%)')

            @chat.on_user_submit
            async def process_chat():
                user_message = chat.user_input()

                if user_message == "Очистить чат":
                    await chat.clear_messages()
                    await chat.append_message_stream('Чат очищен ✅')
                    return

                try:
                    qdrant_client = QdrantClient(url=input.qdrant_url(),
                                                 api_key=input.qdrant_api_key(),)

                    if not qdrant_client.collection_exists(collection_name=input.collection_name()):
                        await chat.append_message_stream('Коллекция не создана ❌')
                        return

                    try:
                        data = processed_data()
                        data = data[['Название региона', 'Данные', 'Опыт работы']].sample(
                            input.filter_data(), random_state=1)
                        loader = DataFrameLoader(
                            data, page_content_column="Данные")
                        documents = loader.load()
                        splitter = RecursiveCharacterTextSplitter()
                        split_documents = splitter.split_documents(documents)
                    except:
                        await chat.append_message('Извините, данные не загружены ❌')
                        return

                    vector_store = QdrantVectorStore(client=qdrant_client,
                                                     collection_name=input.collection_name(),
                                                     embedding=embeddings)
                    ensemble = create_retrievers(vector_store, split_documents)
                except Exception as e:
                    await chat.append_message(f'Извините, произошла ошибка обращения к базе: {str(e)}')
                    return

                model = input.chat_model()
                temperature = input.temp()
                base_url_m = input.base_url1()
                api_key = input.chat_token() or None

                try:
                    llm = ChatOpenAI(model_name=model,
                                     temperature=temperature,
                                     max_tokens=6000,
                                     base_url=base_url_m,
                                     openai_api_key=api_key)

                    llm_chain = (
                        {"context": ensemble | format_docs,
                            "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    response = llm_chain.invoke(user_message)
                    await chat.append_message_stream(response)
                except:
                    await chat.append_message(f'Извините, произошла ошибка обращения к модели, попробуйте снова или измените модель')
