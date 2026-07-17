import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


from Dashboard import DashboardManager #draw, init_app

app = dash.Dash(__name__)



#controls, inputs, parametrize = init_app()

DM = DashboardManager()

# Extract parameter names for table header
#param_names = list(inputs.keys()) if hasattr(inputs, 'keys') else []

app.layout = html.Div([
    html.Div([
        # Left pane (70%)
        html.Div([            
            # Graph
            dcc.Graph(id='live-graph', className='graph-container')
        ], className='left-pane'),
        
        # Right pane (30%)
        html.Div([
            # Top section with buttons and parameter table
            html.Div([
                # Buttons pane
                html.Div([
                    html.Button('⬇ Download Graph', id='download-btn', className='download-btn'),
                    dcc.Clipboard(id="copy_args", style={"fontSize":20}),
                ], className='buttons-pane'),
                
                # Parameter table
                dcc.Markdown(id='params-table', className='params-table', mathjax=True),
                dcc.Download(id='download-image')
            ], className='top-section'),
            
            # Controls
            html.Div(DM.controls, className='controls-container')
        ], className='right-pane')
    ], className='main-layout')
], className='app-container')


@app.callback(Output('live-graph', 'figure'), DM.inputs)
def update_graph(*args):
    formula = DM.parametrize(*args)

    fig = DM.draw(formula)
    fig.update_layout(template='plotly_dark')
    return fig


@app.callback(
    Output('params-table', 'children'),
    DM.inputs
)
def update_table(*args):
    return DM.update_table(*args)



@app.callback(
    Output('download-image', 'data'),
    Input('download-btn', 'n_clicks'),
    State('live-graph', 'figure'),
    prevent_initial_call=True
)
def download_graph(n_clicks, figure):
    """Download graph as image with parameters in filename"""
    if n_clicks is None or figure is None:
        return None
    data =  DM.download_graph(figure)
    return dcc.send_bytes(src = data['content'], filename=data['filename'])
    

@app.callback(
    Output("copy_args", "content"),
    Input("copy_args", "n_clicks"),
    prevent_initial_call=True
    #State("table_cb", "rowData"),
)
def copy_args(_):
    return DM.lastArgs.__repr__()


if __name__ == '__main__':
    app.run(debug=True)