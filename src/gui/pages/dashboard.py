import dash
from dash import html, dcc, callback, Input, Output
import dash_ag_grid as dag
import plotly.express as px
import sys
sys.path.append("..")
import client

dash.register_page(__name__, path="/")

layout = html.Div([
    html.Span([
        html.H1('OWRT Dashboard'),
        dcc.RangeSlider(0, 6, marks={0: '-24 hr', 1: '-12 hr', 2: '-6 hr', 3: '-2 hr', 4: '-30 min', 5: '-3 min', 6: '-5s'}, value=[0, 6], id="timeline"),
    ], className="header-row"),
    html.Div([
        html.Div([
            # chart
            html.Div([
                html.H3("Packet trends", className="card-header"),
                html.Div([
                    dcc.Dropdown(['All', 'Benign', 'Malicious', 'Unknown'], 'All', id='category'),
                    dcc.Tabs(id='trend-tab', value='hist', children=[
                        dcc.Tab(label='Pie', value='pie'),
                        dcc.Tab(label='Hist', value='hist'),
                    ]),
                ], className="row inputrow"),
                html.Div([
                    dcc.Graph(id="trends-graph"),
                ], className="row")
            ], className="card trends"),
            html.Div([
                # packet info
                html.Div([
                    html.H3("Header Data", className="card-header"),
                    html.P([
                        "Packet header: ",
                        html.Span(id="packet-header")
                    ]),
                    html.P("Packet type: Heartbleed"),
                    html.P("Confidence: 71%"),
                    html.Button("Manully Reclassify"),
                    html.Button("Similar Packets"),
                ], className="card"),
                # packet raw data
                html.Div([
                    html.H3("Packet Data", className="card-header"),
                    html.P("0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 ", className="packet-hex")
                ], className="card hex-container")
            ], className="row")
        ], className="col left"),
        # packet table
        html.Div([
            html.Div([
                html.H3("Packets", className="card-header"),
                html.Div([
                    dag.AgGrid(
                        id="packets",
                        columnDefs=[{"field": "Time"}, {"field": "Source"}, {"field": "Destination"}, {"field": "Protocol"}, {"field": "Length"}, {"field": "Info"}],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        columnSize="autoSize",
                        style={"height": "100%"},
                        dashGridOptions={"rowSelection": "single", "rowMultiSelectWithClick": False},
                        getRowId="params.data.No"
                    )
                ], className="tabcont"),
            ], className="card"),
        ], className="col right"),
    ], className="row grow")
], className="pagecontainer")


def packetTrendHist(packets):
    return px.histogram(data_frame=packets, x='Time', color='Protocol').update_layout(margin=dict(l=20, r=20, t=20, b=20))


def packetTrendPie(packets):
    protocolCounts = packets['Protocol'].value_counts()
    return px.pie(data_frame=protocolCounts, names=protocolCounts.keys(), values=protocolCounts.values, color='Protocol').update_layout(margin=dict(l=20, r=20, t=20, b=20))


@callback(
    Output("packets", "rowData"),
    Output("trends-graph", "figure"),  # patch?
    Output("packets", "selectedRows"),
    Output("packets", "dashGridOptions"),
    Input("timeline", "value"),
    Input("trend-tab", "value"),
    Input("category", "value")
)
def updateData(timerange, charttype, category):
    c = client.Client()
    packets = c.getPackets(timerange=timerange, category=category)

    packetTable = packets.to_dict("records")

    figure = None
    if charttype == 'hist':
        figure = packetTrendHist(packets)
    elif charttype == 'pie':
        figure = packetTrendPie(packets)

    # best to hide record ids at some point
    return packetTable, figure, {"ids": ["5"]}, {"rowSelection": "single", "rowMultiSelectWithClick": False}


@callback(
    Output("packet-header", "children"),
    Input("packets", "selectedRows")
)
def inspectPacket(packets):
    if packets:
        return str(packets[0])
    return 'No selection'
