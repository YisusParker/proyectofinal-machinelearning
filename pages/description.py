from dash import html

layout = html.Div(
    [
        html.H2("Descripción del Proyecto"),
        html.P(
            """
        En la era digital, la eficacia de las campañas publicitarias es crucial para el éxito de las marcas en un mercado altamente competitivo.
        Este proyecto explora cómo el análisis de sentimientos, implementado mediante algoritmos de clasificación, puede optimizar las estrategias publicitarias.
        Utilizando técnicas avanzadas de procesamiento de lenguaje natural y machine learning, se analizan grandes volúmenes de datos provenientes de la red social de Twitter,
        donde reseñas de productos y comentarios de usuarios pueden ayudar a identificar patrones y tendencias en las percepciones de los consumidores.
    """
        ),
    ]
)
