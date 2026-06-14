import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

from Capstone.Geometry import pol2cart

def dots_and_arrows(I, X, Y, Ex, Ey, Px, Py, clas, filt, **kwargs):
    Ax, Ay = Ex - X, Ey - Y

    fig = ff.create_quiver(X, Y, Ax, Ay, scale=1, arrow_scale=.05, name='offset',hovertext=I)
    fig.add_trace(go.Scatter(x=Px, y=Py, mode='markers',
                             marker=dict(size=6, color = clas*1, symbol = 'x') ,
                             hovertext = clas,
                             name='window_hits'))
    fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', marker=dict(size=6, color=filt*1),
                              hovertext=I, name='XY'))
    fig.add_trace(go.Scatter(x=Ex, y=Ey, mode='lines', marker=dict(size=6, color=filt*1),
                             hovertext=I, name='ExEy'))
    
    Hx = kwargs['Hx']
    Hy = kwargs['Hy']
    fig.add_trace(go.Scatter(x=Hx, y=Hy, mode='lines', marker=dict(size=6, color=filt*1),
                             name='Hull'))

    fig.update_layout(width=800, height=800)
    # fig.update_xaxes(range=roi['x'])
    # fig.update_yaxes(range=roi['y'])
    fig.show()



def Interactive_polar(df):
    # Interactive Polar plot
    #R upper bound
    Rub = np.ceil(df.R.max()) + 1

    fig = px.line_polar(df, r="R", theta="TD", line_close=True,
                        range_r=[0,Rub], animation_frame="Step", 
                        direction= "counterclockwise", start_angle=0,
                        #color_discrete_sequence=px.colors.sequential.Plasma_r, 
                        #template="plotly_dark",)
                        width=600, height=600
                        )
    fig.show()


def Shape(R, T):
    Rmax = np.max(R) 
    X, Y = pol2cart(R, T)
    mode='lines'
    fig = px.line(x=X, y=Y)
    fig.update_layout(width=500 , height = 500, 
                      yaxis_range = [-Rmax, Rmax], 
                      xaxis_range = [-Rmax, Rmax])
    fig.show()