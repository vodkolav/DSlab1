




from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

from Formulas import formula1
from Signature import analyze_function, dec_scale_2

from dash import dcc
from dash.dependencies import Input, Output
# Convert between Cartesian/Polar coordinates

from datetime import datetime

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

class DashboardManager:


    def draw(self, formula, mode='lines'):
        
        R = formula(T)
        X, Y = pol2cart(R, T)
        fig = make_subplots(rows=2, cols=1, subplot_titles=('R vs T', 'Y vs X'))

        fig.add_trace(go.Scatter(x=X, y=Y, mode=mode, name='Y(X)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=T, y=R, mode=mode, name='R(T)'), row=2, col=1)

        fig.update_yaxes(title_text='Y', row=1, col=1, scaleanchor = 'x', scaleratio=1)

        fig.update_layout(width=600 , height = 1000)
        return fig



    def __init__(self):

        self.formula = formula1

        self.funcparams = analyze_function(self.formula)

        #self.parametrizer = self.emit_parametrizer(self.formula, self.funcparams)

        self.controls = []
        self.inputs = []

        for nm, prm in self.funcparams.items():
            lbl, sldr = self.emit_slider(prm)
            self.controls.append(lbl)
            self.controls.append(sldr)
            self.inputs.append(Input(f'slider-{nm}', 'value'))

        #return controls, inputs, parametrizer
    def emit_slider(self, p: dict):
        Desc, Name, Type, Scl, Min, Max, Step, Def = p.values()
        lbl = dcc.Markdown(f'${Name}$: {Desc}', mathjax=True,)

        if Scl == "Dec":
            ax, vl = dec_scale_2(Min,Max,Step)
            markers = {a: f'{v:.3g}'.format(v) for a,v in zip(ax,vl)}

            sldr = dcc.Slider(id=f'slider-{Name}', updatemode='mouseup',marks= markers,
                    min=Min, max=Max, step=Step, value=Def, 
                    tooltip={"placement": "top", "always_visible": True, "transform": "decScale"})
        else:
            sldr = dcc.Slider(id=f'slider-{Name}', updatemode='mouseup',
                        min=Min, max=Max, step=Step, value=Def )

        return lbl, sldr



    def transform_args(self, args):
        args = list(args)
        for i,(j,k) in enumerate(self.funcparams.items()):
            if k['Scl'] == 'Dec':
                args[i] = 10 ** args[i]
        args = tuple(args)
        return args

    def parametrize(self, *args):
        args = self.transform_args(args)
        self.lastArgs = args
        funct = self.formula(*args)
        return funct


    def update_table(self, *args):
        args = self.transform_args(args)
        """Update parameter table with current values (markdown format)"""
        # Get parameter names and values
        param_names = [v['Name'] for k,v  in self.funcparams.items()]
        #param_names = [ipt.component_id for ipt in inputs]
        
        # Build markdown table
        header = '| ' + ' | '.join(param_names) + ' |'
        separator = '|' + '|'.join(['---'] * len(param_names)) + '|'
        values = '| ' + ' | '.join([f'{value:.3g}' for value in args]) + ' |'
        
        markdown_table = f"{header}\n{separator}\n{values}"
        return markdown_table
    

    def download_graph(self, figure):
        #args = self.transform_args(args)

        try:
            # Get parameter names and values
            param_names = [v['Name'] for k,v  in self.funcparams.items()]
            param_str = '_'.join([f'{name}={val:.2g}' for name, val in zip(param_names, self.lastArgs)])
            
            # Create filename with timestamp and parameters
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'superformula_{param_str}_{timestamp}.png'
            
            # Convert figure to image bytes using plotly
            import plotly.io as pio
            image_bytes = pio.to_image(figure, format='png')
            
            return dict(content=image_bytes, filename=filename)
        except Exception as e:
            print(f'Error downloading image: {e}')
            return None