import dash
from dash import Dash, html
import os
import sys
from client import ClientDataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class GUI:
    def __init__(self):
        self.client = ClientDataLoader()
        self.app = Dash(__name__, use_pages=True)
        self.app.layout = html.Div([
            dash.page_container,
        ])
        if __name__ == '__main__':
            self.app.run(debug=True)


gui = GUI()
