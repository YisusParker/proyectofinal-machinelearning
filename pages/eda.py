from dash import dcc, html, dash_table
import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")
# Asumiendo que ya tienes los datos cargados y procesados
df_sample_head = df.head()
data_cleaned_describe = df.describe()

layout = html.Div(
    [
        html.H2("Análisis Exploratorio de Datos (EDA)"),
        html.Div(
            [
                html.H3("Primeras 5 Filas del DataFrame"),
                html.P(
                    """
        El código muestra las primeras 5 filas del DataFrame `df_sample` utilizando la función `head()`. Este paso es crucial para verificar el contenido y la estructura del DataFrame después del muestreo aleatorio.
        """
                ),
                dash_table.DataTable(
                    data=df_sample_head.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_sample_head.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left"},
                ),
            ],
            className="table-container",
        ),
        html.Div(
            [
                html.H3("Resumen Estadístico Descriptivo"),
                html.P(
                    """
        El código genera un resumen estadístico descriptivo del DataFrame 'data_cleaned'. La función `describe()` proporciona estadísticas resumidas como conteo, media, desviación estándar, mínimos, cuartiles y máximos para las columnas numéricas del DataFrame.
        """
                ),
                dash_table.DataTable(
                    data=data_cleaned_describe.reset_index().to_dict("records"),
                    columns=[
                        {"name": i, "id": i}
                        for i in data_cleaned_describe.reset_index().columns
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left"},
                ),
            ],
            className="table-container",
        ),
        html.H3("Distribución de los Valores de la Columna 'target'"),
        html.P(
            """
    El gráfico de barras muestra la distribución de los valores de la columna `target` en el DataFrame. Este gráfico es útil para visualizar la cantidad de instancias en cada categoría del `target` (0 y 1).
    """
        ),
        html.Img(src="/assets/imgs/diagrama_de_barras_targets.png"),
        html.H3("Actividad a lo Largo del Tiempo"),
        html.P(
            """
    El gráfico de líneas muestra la actividad a lo largo del tiempo, representando el número de tweets por fecha. Este gráfico es útil para visualizar cómo cambia la actividad en el conjunto de datos a lo largo del tiempo.
    """
        ),
        html.Img(src="/assets/imgs/activity_over_time.png"),
        html.H3("Top 20 Usuarios Más Frecuentes"),
        html.P(
            """
    El gráfico de barras horizontal muestra los 20 usuarios más frecuentes en el dataset, indicando el número de tweets realizados por cada usuario.
    """
        ),
        html.Img(src="/assets/imgs/users.png"),
        html.H3("Nube de Palabras para los Textos de Tweets"),
        html.P(
            """
    La nube de palabras muestra una representación visual de las palabras más frecuentes en los textos de tweets.
    """
        ),
        html.Img(src="/assets/imgs/general.png"),
        html.H3("Nube de Palabras para Tweets Positivos"),
        html.P(
            """
    La nube de palabras muestra las palabras más frecuentes en los tweets clasificados como positivos.
    """
        ),
        html.Img(src="/assets/imgs/positive_words.png"),
        html.H3("Nube de Palabras para Tweets Negativos"),
        html.P(
            """
    La nube de palabras muestra las palabras más frecuentes en los tweets clasificados como negativos.
    """
        ),
        html.Img(src="/assets/imgs/negative_words.png"),
        html.H3("Número de Tweets a lo Largo del Tiempo por 'Target'"),
        html.P(
            """
    El gráfico de líneas muestra la evolución del número de tweets a lo largo del tiempo, segmentados por la variable `target`.
    """
        ),
        html.Img(src="/assets/imgs/tweets_over_time_by_target.png"),
    ],
    className="container",
)
