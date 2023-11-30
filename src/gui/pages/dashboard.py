import dash
from dash import html, dcc, callback, Input, Output
import dash_ag_grid as dag
import plotly.express as px
import pandas as pd
import sys
sys.path.append("..")
import clientDataLoader

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
                html.H3("Packet Classifications", className="card-header"),
                html.Div([
                    dcc.Dropdown(['All', 'Benign', 'Malicious', 'Unknown'], 'All', id='category'),
                    dcc.Tabs(id='trend-tab', value='pie', children=[
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
                    html.P([
                        "Packet type: ",
                        html.Span(id="packet-class")
                    ]),
                    html.P([
                        "Probability: ",
                        html.Span(id="packet-conf")
                    ]),
                    # html.Button("Manully Reclassify"),
                    # html.Button("Similar Packets"),
                ], className="card"),
                # packet raw data
                html.Div([
                    html.H3("Payload Data", className="card-header"),
                    html.P("0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 ", className="packet-hex", id="packet-hex")
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
                        columnDefs=[{"field": "pack_id", "hide": True}, {"field": "pack_origin_ip", "headerName": "Source"}, {"field": "srcport", "headerName": "Port (Source)", "filter": "agNumberColumnFilter"}, {"field": "pack_dest_ip", "headerName": "Destination"},
                                    {"field": "destport", "headerName": "Port (Destination)", "filter": "agNumberColumnFilter"}, {"field": "pack_payload", "hide": True}, {"field": "pack_class", "headerName": "Class"}, {"field": "pack_confidence", "headerName": "Probability", "filter": "agNumberColumnFilter"},
                                    {"field": "protocol"}, {"field": "length", "filter": "agNumberColumnFilter"}, {"field": "t_delta", "filter": "agNumberColumnFilter"}, {"field": "ttl", "headerName": "TTL", "filter": "agNumberColumnFilter"}],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        columnSize="sizeToFit",
                        style={"height": "100%"},
                        dashGridOptions={"rowSelection": "single", "rowMultiSelectWithClick": False, "maxBlocksInCache": 3},
                        getRowId="params.data.pack_id",
                        rowModelType="infinite"
                    )
                ], className="tabcont"),
            ], className="card"),
        ], className="col right"),
    ], className="row grow outermost"),
    # dcc.Interval(
    #    id='interval',
    #    interval=500,  # in milliseconds
    #    n_intervals=0
    # )
], className="pagecontainer")


def packetTrendHist(bins):
    binsdf = pd.DataFrame.from_records(bins)
    return px.histogram(data_frame=binsdf, x='bucket', y='ct', color='pack_class').update_layout(margin=dict(l=20, r=20, t=20, b=20))


def packetTrendPie(packets):
    print(packets)
    packetsdf = pd.DataFrame.from_records(packets)
    protocolCounts = packetsdf['ct']  # packetsdf['pack_class'].value_counts()
    print(protocolCounts)
    return px.pie(data_frame=packetsdf, names=packetsdf['pack_class'], values=packetsdf['ct'], color='ct').update_layout(margin=dict(l=20, r=20, t=20, b=20))


@callback(
    Output("trends-graph", "figure"),
    Input("timeline", "value"),
    Input("trend-tab", "value"),
    # Input("category", "value"),
)
def updateChart(timerange, charttype):  # , category):
    c = clientDataLoader.ClientDataLoader()
    figure = None
    if charttype == 'pie':
        counts = c.getClassCounts(timerange=timerange)
        print(counts)
        figure = packetTrendPie(counts)
    elif charttype == 'hist':
        bins = c.getClassBins(timerange=timerange, binct=10)
        figure = packetTrendHist(bins)
    return figure


@callback(
    Output("packets", "getRowsResponse"),  # Output("packets", "rowData"),
    # Output("packets", "selectedRows"),
    # Output("packets", "dashGridOptions"),
    Input("timeline", "value"),
    Input("trend-tab", "value"),
    # Input("category", "value"),
    Input("packets", "getRowsRequest"),
    # Input("interval", "n_intervals")
)
def updateData(timerange, charttype, rowsRequest):  # category
    c = clientDataLoader.ClientDataLoader()
    print(rowsRequest)
    if rowsRequest is None:
        rowsRequest = {'startRow': 0, 'endRow': 100, 'sortModel': [], 'filterModel': {}}

    # switch packet table to infinite (100x speedup)
    packets = c.getPackets(timerange=timerange, category=None, requestData=rowsRequest)
    if len(packets) == 0:
        return [], None

    packetTable = packets  # packets.to_dict("records")

    # best to hide record ids at some point
    return packetTable  # , figure  # , {"ids": []}, {"rowSelection": "single", "rowMultiSelectWithClick": False}


@callback(
    Output("packet-header", "children"),
    Output("packet-hex", "children"),
    Output("packet-class", "children"),
    Output("packet-conf", "children"),
    Input("packets", "selectedRows")
)
def inspectPacket(packets):
    if packets:
        c = clientDataLoader.ClientDataLoader()
        packet = c.getPacket(id=packets[0]["pack_id"])
        byteInts = map(int, str(packet["pack_payload"]).split(","))
        length = packet["length"]
        byteString = ""
        if False: # hex
            byteString = ["{:02x}".format(next(byteInts)) for i in range(0, length)]
        else:
            byteString = "".join([chr(byte) if (byte >= 32 and byte <= 126) else '.' for byte in byteInts])

        headerData = html.Div([f"Source: {packet['pack_origin_ip'].strip()}:{packet['srcport']}",
                              html.Br(),
                              f"Destination: {packet['pack_dest_ip'].strip()}:{packet['destport']}",
                               html.Br(),
                               f"Protcol: {packet['protocol']}",
                               html.Br(),
                               f"Length: {packet['length']}",
                               html.Br(),
                               f"T_delta: {packet['t_delta']}",
                               html.Br(),
                               f"TTL: {packet['ttl']}"
                               ], style={"margin-left": "64px"})

        return headerData, byteString, packet["pack_class"], packet["pack_confidence"]
    return 'No selection', None, None, None
