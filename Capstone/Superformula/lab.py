import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from Dashboard import draw
from Formulas import parametrize
from Signature import analyze_function

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



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
            dcc.Graph(id='live-graph', style={'width': '100%', 'height': '100%'})
        ], style={
            'flex': '0 0 70%',
            'padding': '10px',
            'boxSizing': 'border-box'
        }),
        
        # Right pane (30%)
        html.Div([
            html.Div(controls, style={
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '15px',
                'padding': '10px'
            })
        ], style={
            'flex': '0 0 30%',
            'padding': '10px',
            'boxSizing': 'border-box',
            'backgroundColor': '#f9f9f9',
            'borderLeft': '1px solid #ddd',
            'overflowY': 'auto'
        })
    ], style={
        'display': 'flex',
        'width': '100%',
        'height': '100vh',
        'margin': '0',
        'padding': '0'
    })
], style={
    'width': '100%',
    'height': '100%',
    'margin': '0',
    'padding': '0'
})


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
    # x = np.linspace(0, 10, 100)
    # # Formula: y = b * sin(a*x)
    # y = b * np.sin(a * x)
    
    # fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines')])
    # fig.update_layout(title=f'Formula: y = {b} * sin({a}*x)')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)