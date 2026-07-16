import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from Capstone.Geometry import pol2cart

def dots_and_arrows(I, X, Y, Ex, Ey, Px, Py, clas, filt, **kwargs):
    Ax, Ay = Ex - X, Ey - Y
    isNew = kwargs['IsNew']
    fig = ff.create_quiver(X, Y, Ax, Ay, scale=1, arrow_scale=.05, name='offset',hovertext=I)
    fig.add_trace(go.Scatter(x=Px, y=Py, mode='markers',
                             marker=dict(size=6, color = clas*1, symbol = 'x') ,
                             hovertext = clas,
                             name='window_hits'))
    
    stat = (isNew * 2 + filt)

    colr = {0:"blue", 1: "red", 2:"green", 3:"purple"}

    stat = [colr[i] for i in stat]

    fig.add_trace(go.Scatter(x=X, y=Y, mode='lines', marker=dict(size=4, color=filt*1),
                              hovertext=I, name='XY'))
    fig.add_trace(go.Scatter(x=Ex, y=Ey, mode='markers', marker=dict(size=4, color=stat),
                             hovertext=I, name='ExEy'))
    
    Hx = kwargs['Hx']
    Hy = kwargs['Hy']
    fig.add_trace(go.Scatter(x=Hx, y=Hy, mode='lines', marker=dict(size=4, color=filt*1),
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


def animate(SIM, save_path=None):

    HSdata = SIM.HScurves.results()

    dfData = pd.DataFrame(HSdata)
    dfData.loc[dfData.I == 42,"IsNew"] = 1.0 # so the animation has both colors in the frames
    dfData["IsNew"] = dfData.IsNew.astype(bool)

    dfData["stat"] = (dfData.IsNew * 2 + dfData.filt).astype(str)

    dfData.sort_values(["SimStep", "I", "stat"],inplace=True)

    sddf = pd.DataFrame( {"Hx": SIM.Hx, "Hy": SIM.Hy})
    
    #R upper bound
    Rub = np.ceil(SIM.Hx.max()) 

    fig = px.scatter(dfData, x="X", y="Y", 
                        color="stat", 
                        range_x=[-Rub,Rub], range_y=[-Rub,Rub],
                        hover_data=["I"],                        
                        animation_frame="SimStep", 
                        #name = "Front",
                        # direction= "counterclockwise", start_angle=0,
                        #color_discrete_sequence=px.colors.sequential.Plasma_r, 
                        #template="plotly_dark",)
                        category_orders={"stat": ["0", "1", "2", "3"]},
                        width=700, height=600,
                        render_mode="SVG"
                        )
    hull_trace = go.Scatter(
        x=sddf["Hx"],
        y=sddf["Hy"],
        mode="lines",
        name="Hull",
        line=dict(color="black", width=2),
        hoverinfo="skip",
    )
    fig.add_trace(hull_trace)

    for frame in fig.frames:
        frame.data = list(frame.data) + [
            go.Scatter(
                x=sddf["Hx"],
                y=sddf["Hy"],
                mode="lines",
                name="Hull",
                line=dict(color="black", width=2),
                hoverinfo="skip",
            )
        ]

    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1
        )
    fig.update_traces(marker=dict(size=4))
    if save_path is not None:
        fig.write_html(save_path, auto_play = False )

    else:
        fig.show()
    return dfData


def WebAndPerf(SIM, save_path = None):
    
    HSdata = SIM.HScurves.results()

    dfData = pd.DataFrame(HSdata)
    #dfData.loc[dfData.I == 42,"IsNew"] = 1.0 # so the animation has both colors in the frames
    dfData["IsNew"] = dfData.IsNew.astype(bool)

    dfData["stat"] = (dfData.IsNew * 2 + dfData.filt).astype(str)

    dfData.sort_values(["SimStep", "I", "stat"],inplace=True)

    sddf = pd.DataFrame( {"Hx": SIM.Hx, "Hy": SIM.Hy})


    #R upper bound
    Rub = np.ceil(SIM.Hx.max()*1.01) 

    
    grain = px.line(dfData, x="X", y="Y",
                        # color="stat", 
                        range_x=[-Rub,Rub], range_y=[-Rub,Rub],
                        hover_data=["I"],                        
                        # direction= "counterclockwise", start_angle=0,
                        #color_discrete_sequence=px.colors.sequential.Plasma_r, 
                        #template="plotly_dark",)
                        #category_orders={"stat": ["0", "1", "2", "3"]},
                        # width=700, height=600,
                         render_mode="SVG"
                        )
    

    hull_trace = go.Scatter(
        x=sddf['Hx'],
        y=sddf['Hy'],
        mode='lines',
        name='Hull',
        line=dict(color='black', width=2),
        hoverinfo='skip',
        showlegend=False
    )

    dfSim = pd.DataFrame(SIM.HSsim.results())
    perf = px.line(dfSim, x = 'SimStep', y = 'C', render_mode="SVG",)

    fig = make_subplots(subplot_titles=('Web burned', 'Performance (Steps vs. Curcumfernce)' ), rows=1, cols=2)

    for trace in grain.data:
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(hull_trace, row=1, col=1)

    for trace in perf.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_yaxes(title_text='Y', row=1, col=1, scaleanchor = 'x', scaleratio=1)

    # fig.update_layout(yaxis_range = [-Rmax, Rmax], xaxis_range = [-Rmax, Rmax], **self.conf("layout"))
    
    # 5) Optional layout tweaks
    fig.update_layout(width=1200, height=600, margin=dict(l=50, r=50, t=50, b=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    fig.update_traces(marker=dict(size=1))

    if save_path is not None:
        
        fig.write_html(save_path, auto_play = False )

    else:
        fig.show()
    return dfData


def Shape(R, T):
    Rmax = np.max(R) 
    X, Y = pol2cart(R, T)
    mode='lines'
    fig = px.line(x=X, y=Y)
    fig.update_layout(width=500 , height = 500, 
                      yaxis_range = [-Rmax, Rmax], 
                      xaxis_range = [-Rmax, Rmax])
    fig.show()