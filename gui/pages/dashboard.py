import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.express as px
from outputDataObject import outputDataUpdateObject


dash.register_page(__name__, path="/")

dum = outputDataUpdateObject.dummy()

layout = html.Div([
    html.H1('OWRT Dashboard'),
    html.Div(className='row', children=[
        dcc.Graph(figure=px.pie(values=dum.attacks, names=dum.attacks.keys()), id='packet-type-pie')
    ])
    
])
