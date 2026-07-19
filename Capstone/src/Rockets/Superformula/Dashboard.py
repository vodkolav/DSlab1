
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import numpy as np

from Rockets.Superformula.Formulas import formula1, formula2
from Rockets.Superformula.Signature import analyze_function, dec_scale_2, exp_scale, to_si
from Rockets.Geometry import pol2cart

from dash import dcc
from dash.dependencies import Input, Output
# Convert between Cartesian/Polar coordinates

from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plotly.io.json.config.default_engine = 'orjson'
# orjson is pretty sensitive to types. turns out, you can't have a 
# np.float as dict key. and I've used them in marks for exponential 
# scales in sliders. So need to convert them back to regular float


def fmt(val):
    if isinstance(val, float):
        return f'{val:.3g}' 
    elif isinstance(val,str):
        return val
    else:
        return str(val)

n = 1000

# Theta (radians). To avoid /div0 start from 1 
T = np.linspace(1, np.pi*2 +1 , n)

class DashboardManager:

    @property 
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, key, val):
        self._settings[key] = val


    def conf(self, key):
        # val = self.settings(key)
        (r,c) = (2,1) if self.settings["graph.orientation"] == "vertical" else (1,2)
        (w,h) = (600, 800) if self.settings["graph.orientation"] == "vertical" else (1600, 1200)
        match key:
            case "subplots":
                return {"rows": r, "cols": c, }
            case "trace2":
                return {"row": r, "col": c, }
            case "layout":
                return {"width": w, "height": h}

    def draw(self, formula, mode='lines'):
        
        R = formula(T)
        Rmax = np.max(R) # makes plot ranges invariant to rotations
        X, Y = pol2cart(R, T)
        fig = make_subplots(subplot_titles=('Y vs X', 'R vs T' ), **self.conf("subplots"))

        fig.add_trace(go.Scatter(x=X, y=Y, mode=mode, name='X(T), Y(T)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=T, y=R, mode=mode, name='R(T)'), **self.conf("trace2"))

        fig.update_yaxes(title_text='Y', row=1, col=1, scaleanchor = 'x', scaleratio=1)

        fig.update_layout(yaxis_range = [-Rmax, Rmax], xaxis_range = [-Rmax, Rmax], **self.conf("layout"))
        return fig



    def __init__(self, settings = None):

        self._settings = {"graph.orientation": "horizontal"}

        self.formula = formula1

        self.funcparams = analyze_function(self.formula)

        #self.parametrizer = self.emit_parametrizer(self.formula, self.funcparams)

        self.lastArgs = {}

        self.controls = []
        self.inputs = []

        for nm, prm in self.funcparams.items():
            lbl, sldr, inpt = self.emit_control(prm)
            self.controls.append(lbl)
            self.controls.append(sldr)
            self.inputs.append(inpt)

        #return controls, inputs, parametrizer
    def emit_control(self, p: dict):
        Desc, Name, Type, Scl, Opts, Min, Max, Step, Def = p.values()
        lbl = dcc.Markdown(f'${Name}$: {Desc}', mathjax=True,)

        if Scl == "Choice":
            sldr = dcc.Dropdown(id=f'dropdown-{Name}', options=Opts, value = Def, clearable=False, searchable=False )
            inpt = Input(f'dropdown-{Name}', 'value')

        elif Scl == "Exp":
            ax, vl = exp_scale(Min,Max,20)
            markers = {float(a): to_si(v) for a,v in zip(ax,vl)}

            sldr = dcc.Slider(id=f'slider-{Name}', updatemode='mouseup',marks= markers,
                    min=ax[0], max=ax[-1], step=Step, value=Def,
                    tooltip={"placement": "top", "always_visible": True, "transform": "decScale"})
            inpt = Input(f'slider-{Name}', 'value')
        else:
            sldr = dcc.Slider(id=f'slider-{Name}', updatemode='mouseup',
                        min=Min, max=Max, step=Step, value=Def )
            inpt = Input(f'slider-{Name}', 'value')

        return lbl, sldr, inpt



    def transform_args(self, args):
        args = list(args)
        for i,(j,k) in enumerate(self.funcparams.items()):
            if k['Scl'] == 'Exp':
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
        param_names = [f"${v['Name']}$" for k,v  in self.funcparams.items()]
        #param_names = [ipt.component_id for ipt in inputs]
        
        # Build markdown table
        header = '| ' + ' | '.join(param_names) + ' |'
        separator = '|' + '|'.join(['---'] * len(param_names)) + '|'
        values = '| ' + ' | '.join([fmt(value) for value in args]) + ' |'
        
        markdown_table = f"{header}\n{separator}\n{values}"
        return markdown_table
    

    def download_graph(self, figure):
        #args = self.transform_args(args)

        try:
            # Get parameter names and values
            param_names = [v['Name'] for k,v  in self.funcparams.items()]
            param_str = ' '.join([f'{name}={fmt(val)}' for name, val in zip(param_names, self.lastArgs)])
            
            # Create filename with timestamp and parameters
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{param_str}_{timestamp}.png'
            
            # Convert figure to image bytes using plotly
            import plotly.io as pio
            image_bytes = pio.to_image(figure, format='png')
            
            return dict(content=image_bytes, filename=filename)
        except Exception as e:
            logger.error(f'Error downloading image: {e}')
            return None