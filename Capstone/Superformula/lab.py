import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from Dashboard import draw
from Formulas import parametrize
from Signature import analyze_function, dec_scale_2

app = dash.Dash(__name__)



def emit_slider(p: dict):
    Desc, Name, Type, Scl, Min, Max, Step, Def = p.values()
    lbl = dcc.Markdown(f'${Name}$: {Desc}', mathjax=True,)

    if Scl == "Dec":
        ax, vl = dec_scale_2(Min,Max,Step)
        markers = {a:  f'{v:.3g}'.format(v) for a,v in zip(ax,vl)}

        sldr = dcc.Slider(id=f'slider-{Name}', updatemode='drag',marks= markers,
                min=Min, max=Max, step=Step, value=Def, 
                tooltip={"placement": "top", "always_visible": True, "transform": "decScale"})
    else:
        sldr = dcc.Slider(id=f'slider-{Name}', updatemode='drag',
                      min=Min, max=Max, step=Step, value=Def )
        
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


@app.callback(Output('live-graph', 'figure'), callbcks)
def update_graph(*args):
    formula = parametrize(*args)

    fig = draw(formula)
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run(debug=True)