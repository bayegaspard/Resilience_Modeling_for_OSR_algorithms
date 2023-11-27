import dash
from dash import Dash, html
import os
import sys
from clientDataLoader import ClientDataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class GUI:
    def __init__(self):
        self.client = ClientDataLoader()
        self.app = Dash(__name__, use_pages=True)
        self.app.layout = html.Div([
            dash.page_container,
        ])

    def run(self):
        self.app.run(debug=True)
