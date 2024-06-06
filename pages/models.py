from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Resultados de los modelos
xgb_report = {
    '0': {'precision': 0.7865615794920786, 'recall': 0.6222446151908301, 'f1-score': 0.6948204929849854, 'support': 15878.0},
    '1': {'precision': 0.6914450331807191, 'recall': 0.8337054955960799, 'f1-score': 0.7559404966114563, 'support': 16122.0},
    'accuracy': 0.72878125,
    'macro avg': {'precision': 0.7390033063363989, 'recall': 0.727975055393455, 'f1-score': 0.7253804947982209, 'support': 32000.0},
    'weighted avg': {'precision': 0.7386406745035868, 'recall': 0.72878125, 'f1-score': 0.7256135148120468, 'support': 32000.0}
}

ridge_report = {
    '0': {'precision': 0.7842955324182997, 'recall': 0.7419714848207456, 'f1-score': 0.7625466765040386, 'support': 159494.0},
    '1': {'precision': 0.7566479218037644, 'recall': 0.7972225337370565, 'f1-score': 0.7764054863342222, 'support': 160506.0},
    'accuracy': 0.769684375,
    'macro avg': {'precision': 0.7704717271110321, 'recall': 0.7695970092789011, 'f1-score': 0.7694760814191304, 'support': 320000.0},
    'weighted avg': {'precision': 0.7704280093267478, 'recall': 0.769684375, 'f1-score': 0.7694979956621745, 'support': 320000.0}
}

knn_report = {
    '0': {'precision': 0.5875356573030992, 'recall': 0.7653356845950372, 'f1-score': 0.6647520582040972, 'support': 15878.0},
    '1': {'precision': 0.6707608023327737, 'recall': 0.4708472894181863, 'f1-score': 0.5533000473778199, 'support': 16122.0},
    'accuracy': 0.61696875,
    'macro avg': {'precision': 0.6291482298179365, 'recall': 0.6180914870066118, 'f1-score': 0.6090260527909586, 'support': 32000.0},
    'weighted avg': {'precision': 0.6294655256833621, 'recall': 0.61696875, 'f1-score': 0.6086011419996834, 'support': 32000.0}
}

rf_report = {
    '0': {'precision': 0.7660008618285975, 'recall': 0.7578843091276161, 'f1-score': 0.7619209701920592, 'support': 159494.0},
    '1': {'precision': 0.7619176798441392, 'recall': 0.7699400645458737, 'f1-score': 0.7659078654610135, 'support': 160506.0},
    'accuracy': 0.76393125,
    'macro avg': {'precision': 0.7639592708363683, 'recall': 0.7639121868367449, 'f1-score': 0.7639144178265364, 'support': 320000.0},
    'weighted avg': {'precision': 0.7639528143048554, 'recall': 0.76393125, 'f1-score': 0.7639207221046803, 'support': 320000.0}
}

# Crear una tabla comparativa
comparison_table = pd.DataFrame({
    'Model': ['XGBoost', 'Ridge', 'KNN', 'Random Forest'],
    'Accuracy': [xgb_report['accuracy'], ridge_report['accuracy'], knn_report['accuracy'], rf_report['accuracy']],
    'Precision (0)': [xgb_report['0']['precision'], ridge_report['0']['precision'], knn_report['0']['precision'], rf_report['0']['precision']],
    'Recall (0)': [xgb_report['0']['recall'], ridge_report['0']['recall'], knn_report['0']['recall'], rf_report['0']['recall']],
    'F1-score (0)': [xgb_report['0']['f1-score'], ridge_report['0']['f1-score'], knn_report['0']['f1-score'], rf_report['0']['f1-score']],
    'Precision (1)': [xgb_report['1']['precision'], ridge_report['1']['precision'], knn_report['1']['precision'], rf_report['1']['precision']],
    'Recall (1)': [xgb_report['1']['recall'], ridge_report['1']['recall'], knn_report['1']['recall'], rf_report['1']['recall']],
    'F1-score (1)': [xgb_report['1']['f1-score'], ridge_report['1']['f1-score'], knn_report['1']['f1-score'], rf_report['1']['f1-score']]
})

# Crear gráfica de matriz de calor
fig_heatmap = px.imshow(comparison_table.set_index('Model').T, text_auto=True, color_continuous_scale='RdBu', aspect="auto", title="Comparación de Modelos")

# Crear la gráfica comparativa
comparison_table.set_index('Model', inplace=True)
comparison_table = comparison_table.transpose()
metrics_to_plot = ['Accuracy', 'Precision (0)', 'Recall (0)', 'F1-score (0)', 'Precision (1)', 'Recall (1)', 'F1-score (1)']

fig_comparison = go.Figure()
for metric in metrics_to_plot:
    fig_comparison.add_trace(go.Scatter(x=comparison_table.columns, y=comparison_table.loc[metric], mode='lines+markers', name=metric))

fig_comparison.update_layout(
    title='Comparación de Modelos',
    xaxis_title='Modelos',
    yaxis_title='Puntuaciones',
    legend_title='Métricas',
    template='plotly_white'
)

# Definir las matrices de confusión manualmente usando los valores proporcionados
def generate_confusion_matrix(report):
    support_0 = report['0']['support']
    support_1 = report['1']['support']
    recall_0 = report['0']['recall']
    recall_1 = report['1']['recall']
    fn_0 = support_0 - (recall_0 * support_0)  # Falsos Negativos para la clase 0
    fn_1 = support_1 - (recall_1 * support_1)  # Falsos Negativos para la clase 1

    tp_0 = recall_0 * support_0  # Verdaderos Positivos para la clase 0
    tp_1 = recall_1 * support_1  # Verdaderos Positivos para la clase 1

    fp_0 = support_1 - fn_1  # Falsos Positivos para la clase 0
    fp_1 = support_0 - fn_0  # Falsos Positivos para la clase 1

    cm = np.array([[tp_0, fp_0],
                   [fn_1, tp_1]])
    return cm.astype(int)

models = {
    'XGBoost': xgb_report,
    'Ridge': ridge_report,
    'KNN': knn_report,
    'Random Forest': rf_report
}

# Crear gráficos de matrices de confusión
confusion_matrices = []
for model_name, report in models.items():
    cm = generate_confusion_matrix(report)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        hoverongaps=False,
        colorscale='Blues',
        showscale=False
    ))
    fig_cm.update_layout(title=f'Confusion Matrix - {model_name}', xaxis_title='Predicted', yaxis_title='Actual')
    confusion_matrices.append(dcc.Graph(figure=fig_cm))

# Layout de Dash para la sección de modelos
layout = html.Div([
    html.H2("Comparación de Modelos"),

    html.H3("Tabla Comparativa de Modelos"),
    dash_table.DataTable(
        data=comparison_table.transpose().to_dict('records'),
        columns=[{"name": i, "id": i} for i in comparison_table.transpose().columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    ),

    html.H3("Gráfica de Comparación de Métricas"),
    dcc.Graph(figure=fig_comparison),

    html.H4("Interpretación del Gráfico Comparativo de Modelos"),
    html.P([
        "El gráfico comparativo de modelos muestra las puntuaciones de varias métricas de evaluación (Accuracy, Precision, Recall, F1-score) para los modelos XGBoost, Ridge, KNN, y Random Forest. Cada línea en el gráfico representa una métrica diferente, permitiendo una visualización clara de las diferencias de rendimiento entre los modelos."
    ]),
    html.H5("Observaciones Clave:"),
    html.Ul([
        html.Li([html.Strong("Accuracy"), ": Los modelos Ridge y Random Forest tienen las puntuaciones de accuracy más altas y similares, seguidos por XGBoost, mientras que KNN tiene la puntuación más baja."]),
        html.Li([html.Strong("Precision (0)"), ": XGBoost tiene la precisión más alta para la clase negativa, seguido de cerca por Ridge y Random Forest. KNN tiene la precisión más baja en esta métrica."]),
        html.Li([html.Strong("Recall (0)"), ": KNN destaca con el recall más alto para la clase negativa, lo que indica su capacidad para identificar correctamente la mayoría de los comentarios negativos. Sin embargo, otros modelos tienen un recall más equilibrado y consistente."]),
        html.Li([html.Strong("F1-score (0)"), ": Ridge y Random Forest tienen puntuaciones de F1-score muy similares y altas para la clase negativa, lo que indica un buen equilibrio entre precisión y recall. XGBoost sigue con una puntuación moderada, mientras que KNN tiene la más baja."]),
        html.Li([html.Strong("Precision (1)"), " y ", html.Strong("Recall (1)"), ": Los modelos muestran un rendimiento relativamente equilibrado en estas métricas para la clase positiva, con XGBoost teniendo un recall notablemente alto para la clase positiva."]),
        html.Li([html.Strong("F1-score (1)"), ": Las puntuaciones son bastante consistentes entre Ridge, Random Forest y XGBoost, con KNN nuevamente siendo el más bajo."])
    ]),

    html.H3("Gráfica de Matriz de Calor"),
    dcc.Graph(figure=fig_heatmap),

    html.H4("Interpretación de la Matriz de Comparación de Modelos"),
    html.P([
        "La matriz de comparación de modelos proporciona una visualización detallada de las métricas de evaluación (Accuracy, Precision, Recall, F1-score) para los modelos XGBoost, Ridge, KNN y Random Forest. Cada celda en la matriz representa el valor de una métrica específica para un modelo particular, con colores que indican la magnitud de los valores."
    ]),
    html.H5("Observaciones Clave:"),
    html.Ul([
        html.Li([html.Strong("Accuracy"), ": Ridge y Random Forest tienen las puntuaciones más altas, seguidos de XGBoost y KNN."]),
        html.Li([html.Strong("Precision (0)"), ": XGBoost tiene la precisión más alta, seguido por Ridge y Random Forest. KNN tiene la precisión más baja."]),
        html.Li([html.Strong("Recall (0)"), ": KNN tiene el recall más alto para la clase negativa, lo que indica que es muy efectivo para identificar los comentarios negativos. Sin embargo, su precisión es baja."]),
        html.Li([html.Strong("F1-score (0)"), ": Ridge y Random Forest tienen los F1-scores más altos, seguidos por XGBoost y KNN."]),
        html.Li([html.Strong("Precision (1)"), " y ", html.Strong("Recall (1)"), ": XGBoost tiene el recall más alto para la clase positiva, mientras que Ridge y Random Forest tienen un rendimiento más equilibrado en ambas métricas."]),
        html.Li([html.Strong("F1-score (1)"), ": Los F1-scores son más altos para Ridge, Random Forest y XGBoost, con KNN siendo el más bajo."])
    ]),

    html.H3("Matrices de Confusión"),
    *confusion_matrices,

    html.H4("Interpretación de las Matrices de Confusión"),
    html.H5("Modelo XGBoost"),
    html.P([
        "- ", html.Strong("True Negatives (Predicted Negative - Actual Negative)"), ": 9880",
        html.Br(),
        "- ", html.Strong("False Positives (Predicted Positive - Actual Negative)"), ": 13441",
        html.Br(),
        "- ", html.Strong("False Negatives (Predicted Negative - Actual Positive)"), ": 2681",
        html.Br(),
        "- ", html.Strong("True Positives (Predicted Positive - Actual Positive)"), ": 13441",
        html.Br(),
        "La matriz de confusión para el modelo XGBoost muestra que el modelo tiene una cantidad significativa de falsos positivos (13441) y verdaderos negativos (9880). Esto indica que el modelo tiene dificultades para clasificar correctamente las muestras negativas."
    ]),
    html.H5("Modelo Ridge"),
    html.P([
        "- ", html.Strong("True Negatives (Predicted Negative - Actual Negative)"), ": 118340",
        html.Br(),
        "- ", html.Strong("False Positives (Predicted Positive - Actual Negative)"), ": 127959",
        html.Br(),
        "- ", html.Strong("False Negatives (Predicted Negative - Actual Positive)"), ": 32547",
        html.Br(),
        "- ", html.Strong("True Positives (Predicted Positive - Actual Positive)"), ": 127959",
        html.Br(),
        "La matriz de confusión para el modelo Ridge muestra un alto número de falsos positivos (127959) y verdaderos negativos (118340). Aunque el modelo tiene un mejor desempeño en términos de verdaderos positivos, sigue teniendo una alta tasa de falsos positivos."
    ]),
    html.H5("Modelo K-Nearest Neighbors (KNN)"),
    html.P([
        "- ", html.Strong("True Negatives (Predicted Negative - Actual Negative)"), ": 12152",
        html.Br(),
        "- ", html.Strong("False Positives (Predicted Positive - Actual Negative)"), ": 7591",
        html.Br(),
        "- ", html.Strong("False Negatives (Predicted Negative - Actual Positive)"), ": 8531",
        html.Br(),
        "- ", html.Strong("True Positives (Predicted Positive - Actual Positive)"), ": 7591",
        html.Br(),
        "La matriz de confusión para el modelo KNN muestra un balance entre los verdaderos negativos (12152) y los verdaderos positivos (7591), pero tiene un número considerable de falsos negativos (8531) y falsos positivos (7591)."
    ]),
    html.H5("Modelo Random Forest"),
    html.P([
        "- ", html.Strong("True Negatives (Predicted Negative - Actual Negative)"), ": 120878",
        html.Br(),
        "- ", html.Strong("False Positives (Predicted Positive - Actual Negative)"), ": 123580",
        html.Br(),
        "- ", html.Strong("False Negatives (Predicted Negative - Actual Positive)"), ": 36926",
        html.Br(),
        "- ", html.Strong("True Positives (Predicted Positive - Actual Positive)"), ": 123580",
        html.Br(),
        "La matriz de confusión para el modelo Random Forest muestra un alto número de verdaderos negativos (120878) y verdaderos positivos (123580), pero también tiene una cantidad considerable de falsos positivos (123580) y falsos negativos (36926)."
    ]),

    html.H4("Conclusión"),
    html.P([
        "Para concluir finalmente, si el objetivo principal es minimizar los falsos negativos (es decir, clasificar correctamente los comentarios negativos), el modelo ", html.Strong("Ridge"), " parece ser la mejor opción, ya que tiene un menor número de falsos negativos en comparación con los otros modelos. Sin embargo, todos los modelos tienen sus fortalezas y debilidades, y la elección final del modelo dependerá del contexto específico y de la tolerancia a los falsos positivos y negativos."
    ]),
    html.P([
        "En la era digital, la eficacia de las campañas publicitarias es crucial para el éxito de las marcas en un mercado altamente competitivo. Este artículo explora cómo el análisis de sentimientos, implementado mediante algoritmos de clasificación, puede optimizar las estrategias publicitarias. Utilizando técnicas avanzadas de procesamiento de lenguaje natural y machine learning, se analizan grandes volúmenes de datos provenientes de la red social de Twitter, donde reseñas de productos y comentarios de usuarios pueden ayudar a identificar patrones y tendencias en las percepciones de los consumidores."
    ]),
    html.P([
        "Al centrarse en la correcta clasificación de comentarios negativos, se puede identificar mejor las áreas problemáticas y abordar de manera proactiva las preocupaciones de los consumidores, mejorando así la reputación de la marca y la satisfacción del cliente. En este contexto, el modelo ", html.Strong("Ridge"), " destaca por su capacidad para minimizar los falsos negativos, permitiendo a las empresas detectar con mayor precisión los comentarios negativos y ajustar sus estrategias de marketing en consecuencia."
    ]),
    html.P([
        "Este estudio destaca el potencial del análisis de sentimientos como una herramienta poderosa para mejorar la efectividad y el retorno de inversión de las campañas publicitarias, proporcionando a los profesionales de marketing un enfoque basado en datos para abordar las necesidades y preferencias de los consumidores. La capacidad de clasificar con precisión los comentarios negativos es particularmente valiosa, ya que permite a las empresas mitigar los impactos negativos y reforzar los aspectos positivos de sus estrategias de marketing. Al aplicar estos modelos de análisis de sentimientos, las empresas pueden lograr una segmentación más precisa del mercado, una personalización de los mensajes publicitarios y una toma de decisiones estratégicas más informada."
    ])
], className='container')
