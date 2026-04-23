import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from Dashboard import draw
from Formulas import parametrize
from Signature import analyze_function

app = dash.Dash(__name__)



def emit_slider(p: dict):
    Desc, Name, Type, Min, Max, Step, Def = p.values()
    lbl = html.Label(f'{Desc} ({Name})')
    sldr = dcc.Slider(id=f'slider-{Name}', min=Min, 
               max=Max, step=Step, value=Def, updatemode='drag')
    return lbl, sldr

def init_app():
    funcparams = analyze_function(parametrize)

    controls = []
    callbcks = []

    for nm, prm in funcparams.items():
        lbl, sldr = emit_slider(prm)
        controls.append(lbl)
        controls.append(sldr)
        callbcks.append(Input(f'slider-{nm}', 'value'))

    return controls, callbcks


controls, callbcks = init_app()

app.layout = html.Div([
    html.Div([
        # Left pane (70%)
        html.Div([
            dcc.Graph(id='live-graph')
        ], className='left-pane'),
        
        # Right pane (30%)
        html.Div([
            html.Div(controls, className='controls-container')
        ], className='right-pane')
    ], className='main-layout')
], className='app-container')


# [   dcc.Graph(id='live-graph'),
#     html.Label('Offset (o)'),
#     dcc.Slider(id='slider-o', min=-10, max=10, step=0.5, value=1, updatemode='drag'),
#     html.Label('Amplitude (a)'),
#     dcc.Slider(id='slider-a', min=1, max=10, step=0.1, value=1, updatemode='drag'),
#     html.Label('Amplitude2 (b)'),
#     dcc.Slider(id='slider-b', min=1, max=5, step=0.1, value=1, updatemode='drag' )
# ]


@app.callback(Output('live-graph', 'figure'),
              callbcks)
            #   [Input('slider-o', 'value'),
            #    Input('slider-a', 'value'),
            #    Input('slider-b', 'value')])

def update_graph(*args):
    formula = parametrize(*args)

    fig = draw(formula)
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)