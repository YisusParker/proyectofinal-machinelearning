import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Flask

# Inicializar el servidor Flask
server = Flask(__name__)

# Inicializar la aplicación Dash
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css",
        "/assets/styles.css",
    ],
)
app.title = "Proyecto Machine Learning"

# Importar los módulos de las páginas
from pages import index, eda, models, description

# Definir el layout principal con un menú de navegación
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Nav(
            [
                dcc.Link("Descripción", href="/", className="nav-link"),
                dcc.Link("EDA", href="/eda", className="nav-link"),
                dcc.Link("Modelos", href="/models", className="nav-link"),
            ],
            className="navbar navbar-expand-lg navbar-dark bg-primary",
        ),
        html.Div(id="page-content", className="container"),
    ]
)

# Callback para actualizar la página según la URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/eda":
        return eda.layout
    elif pathname == "/models":
        return models.layout
    elif pathname == "/description":
        return description.layout
    else:
        return index.layout

if __name__ == "__main__":
    app.run_server(debug=True)
