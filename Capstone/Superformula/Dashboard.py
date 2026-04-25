




from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from Formulas import formula1
from Signature import analyze_function, dec_scale_2

from dash import dcc
from dash.dependencies import Input, Output
# Convert between Cartesian/Polar coordinates

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def magn(x, y):
    # magnitude of vector
    return np.sqrt(x**2 + y**2)

def rad2deg(rad):
    # convert radians to degrees
    return rad * 180 / np.pi

n = 1000

# Theta (radians). To avoid /div0 start from 1 
T = np.linspace(1, np.pi*2 +1 , n)

def draw(formula, mode='lines'):
    
    R = formula(T)
    X, Y = pol2cart(R, T)
    fig = make_subplots(rows=2, cols=1, subplot_titles=('R vs T', 'Y vs X'))

    fig.add_trace(go.Scatter(x=X, y=Y, mode=mode, name='Y(X)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=T, y=R, mode=mode, name='R(T)'), row=2, col=1)

    fig.update_yaxes(title_text='Y', row=1, col=1, scaleanchor = 'x', scaleratio=1)

    fig.update_layout(width=600 , height = 1000)
    return fig

def init_app():
    funcparams = analyze_function(formula1)

    parametrizer = emit_parametrizer(formula1, funcparams)

    controls = []
    inputs = []

    for nm, prm in funcparams.items():
        lbl, sldr = emit_slider(prm)
        controls.append(lbl)
        controls.append(sldr)
        inputs.append(Input(f'slider-{nm}', 'value'))

    return controls, inputs, parametrizer


def emit_slider(p: dict):
    Desc, Name, Type, Scl, Min, Max, Step, Def = p.values()
    lbl = dcc.Markdown(f'${Name}$: {Desc}', mathjax=True,)

    if Scl == "Dec":
        ax, vl = dec_scale_2(Min,Max,Step)
        markers = {a: f'{v:.3g}'.format(v) for a,v in zip(ax,vl)}

        sldr = dcc.Slider(id=f'slider-{Name}', updatemode='drag',marks= markers,
                min=Min, max=Max, step=Step, value=Def, 
                tooltip={"placement": "top", "always_visible": True, "transform": "decScale"})
    else:
        sldr = dcc.Slider(id=f'slider-{Name}', updatemode='drag',
                      min=Min, max=Max, step=Step, value=Def )

    return lbl, sldr


def emit_parametrizer(formula, pars):

    def parametrize(*args):
        args = list(args)
        for i,(j,k) in enumerate(pars.items()):
            if k['Scl'] == 'Dec':
                args[i] = 10 ** args[i]
        args = tuple(args)
        funct = formula(*args)
        return funct
    
    return parametrize