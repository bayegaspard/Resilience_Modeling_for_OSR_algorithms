import dash
from dash import Dash, html
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


app = Dash(__name__, use_pages=True)
app.layout = html.Div([
    html.Div(children='Hello World'),
    dash.page_container,
])

if __name__ == '__main__':
    app.run(debug=True)