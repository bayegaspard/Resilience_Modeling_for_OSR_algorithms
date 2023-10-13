import dash
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd
from outputDataObject import outputDataUpdateObject


dash.register_page(__name__, path="/")

dum = outputDataUpdateObject.dummy()


df = px.data.tips()

layout = html.Div([
    html.Span([
        html.H1('OWRT Dashboard'),
        dcc.RangeSlider(0, 24, marks={0: '-24 hours', 6: '-18 hours', 12: '-12 hours', 18: '-6 hours', 24: 'Now'}, value=[0, 24]),
    ], className="header-row"),
    html.Div(className='dashboard-card-cont', children=[
        html.Div(className='card histcont', children=[
            html.H2(className='card-header', children='Histogram of packet types'),
            dcc.Graph(figure=px.histogram(px.data.tips(), x="tip", color="smoker").update_layout(margin=dict(l=20, r=20, t=20, b=20)), id='packet-type-pie')
        ]),
        html.Div(className='card piecont', children=[
            html.H2(className='card-header', children='Packet makeup'),
            dcc.Graph(figure=px.pie(values=outputDataUpdateObject.dummy().attacks, names=outputDataUpdateObject.dummy().attacks.keys()).update_layout(margin=dict(l=20, r=20, t=20, b=20)), id='packet-type-pie')
        ]),
        html.Div(className='card tabcont', children=[
            html.H2(className='card-header', children='Suspicious packets'),
            dash_table.DataTable(data=df.to_dict('records'), columns=[{"name": i, "id": i} for i in df.columns], sort_action='native', filter_action='native', page_action='native', page_size=6, id='packet-table')
        ]),
        html.Div(className='card healthcont', children=[
            html.H2(className='card-header', children='Model health'),
            html.Div(className='row', children=[
                dcc.Graph(figure=px.line(px.data.tips().sort_values(by="total_bill"), x="total_bill", y="tip").update_layout(margin=dict(l=20, r=20, t=20, b=20)), id='packet-type-pie'),
                html.Div(children=[
                    html.P(children='Uptime: 300h 26m'),
                    html.P(children='Time since retrain: 300h 26m'),
                    html.P(children='Unknown rate: ~596/hour'),
                    html.P(children='Unknown makeup: 90%'),
                    html.P(children='Unknown growth: ~10/hour'),
                    html.Hr(),
                    html.P(children='Retraining progress: 0%'),
                    html.Button('Retrain model', className='cardbutton'),
                ], className='col')
            ])
        ])
    ]),
    html.Div(
        className='card',
        id='packet-viewer',
        children=[
            html.H3("192.168.8.23 -> 192.168.3.17", className="card-header"),
            dcc.Input(type="text", placeholder="Packet Identification"),
            html.P("Confidence: 0%"),
            html.Div(className="row nowrap", children=[
                html.Div(className="col", children=[
                    html.P("Packet header info"),
                ]),
                html.Button('<-'),
                html.Span("0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000", className="packet-hex"),
                html.Button('->'),
                html.Div(className="col", children=[
                    html.P("Predictions"),
                ]),
            ]),
        ],
    ),
])


@dash.callback(
    dash.Output(component_id='packet-viewer', component_property='className'),
    dash.Input(component_id='packet-table', component_property='selected_rows')
)
def examinePacket():
    return 