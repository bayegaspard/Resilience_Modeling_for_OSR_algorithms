import dash
from dash import Dash, html
import os
import sys
from client import Client
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class GUI:
    def __init__(self):
        self.client = Client()
        self.app = Dash(__name__, use_pages=True)
        self.app.layout = html.Div([
            dash.page_container,
        ])
        if __name__ == '__main__':
            self.app.run(debug=True)


gui = GUI()
