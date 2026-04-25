import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from Dashboard import draw, init_app

app = dash.Dash(__name__)



controls, inputs, parametrize = init_app()

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


@app.callback(Output('live-graph', 'figure'), inputs)
def update_graph(*args):
    formula = parametrize(*args)

    fig = draw(formula)
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run(debug=True)