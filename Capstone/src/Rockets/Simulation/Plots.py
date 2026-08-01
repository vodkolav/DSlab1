import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from Rockets.Geometry import pol2cart

def dots_and_arrows(SIM, filtr, save_path = None, **kwargs):   

    dfC, color_map = preproc_curves_data(SIM, dopad = False, **filtr)

    dfC["statCol"] = dfC.status.map(color_map)

    dfI = SIM.HSintersections.results(**filtr)

    dfI = pd.DataFrame(dfI)

    Ax, Ay = dfC.Ex - dfC.X, dfC.Ey - dfC.Y

    fig = ff.create_quiver(dfC.X, dfC.Y, Ax, Ay, hovertext=dfC.I, 
                           scale=1, arrow_scale=.05, name='offset')

    fig.add_trace(go.Scatter(x=dfI.Px, y=dfI.Py,  hovertext = dfI.clas,
                             mode='markers',name='intersections',
                             marker=dict(size=6, color = dfI.clas, symbol = 'x')
                             ))

    fig.add_trace(go.Scatter(x=dfC.X, y=dfC.Y, hovertext=dfC.I, name='XY',
                             marker=dict(size=4), # , color=dfC.statCol
                             mode='lines' 
                            ))

    fig.add_trace(go.Scatter(x=dfC.Ex, y=dfC.Ey, mode='lines',  name='ExEy-line'))    

    fig.add_trace(go.Scatter(x=dfC.Ex, y=dfC.Ey, mode='markers', 
                             marker=dict(size=4, color=dfC.statCol), 
                             hovertext=dfC.I, name='ExEy'))

    fig.update_layout(shapes = [casing(SIM.hr)])

    fig.update_layout(width=800, height=800)

    if save_path is not None:
        fig.write_html(save_path, auto_play = False, )

    else:
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

def pad(df: pd.DataFrame, cols = ['frame', 'category'] ):
    """Workaround for plotly bug. 
    When creating the px.scatter animation, 
    the df does not have all the possible values of "category" 
    within the values for the first frame. 
    thus only those that are in the first frame end up in the plot legend. 
    and it stays like this for the whole duration of the animation, 
    even though points with color corresponding to these other 
    categories do appear on the animated plot. 
    but not in the legend. 

    Args:
        df (DataFrame): the DataFrame to pad
        cols (list, optional): columns that together compose a unique multiindex. Defaults to ['frame', 'category'].

    Returns:
        DataFrame: the padded DataFrame
    """

    all_cols = [df[c].unique() for c in cols]

    mux = pd.MultiIndex.from_product(all_cols, names=cols)

    # 2. Reindex dataframe to fill missing slots with NaN/None values
    padded_df = df.set_index(cols)
    padded_df = padded_df.reindex(mux)
    padded_df = padded_df.reset_index()

    return padded_df


def preproc_sim_data(HSsim):

    dfSim = pd.DataFrame(HSsim)

    return dfSim


def preproc_curves_data(HScurves, dopad = False, **kwargs):

    HScurves = pd.DataFrame(HScurves)

    HScurves["stat"] = HScurves.IsNew * 1  + HScurves.filt*2 + (~HScurves.A) * 4 

    if dopad:
        HScurves = pad(HScurves,["SimStep", "I", "stat" ])

    cat_map = {
    0: "0old",
    1: "1new",
    2: "2die",
    3: "3new_die",
    4: "4fin ",
    5: "5new_fin",
    6: "6die_fin",
    7: "7new_die_fin"
    }

    HScurves["status"] = HScurves.stat.map(cat_map)

    color_map = {
    "0old": "#124FC0",
    "1new": '#00CC96',
    "2die": '#FF4B4B',
    "3new_die": "#BF0BEC",
    "4fin ": "#E69112",
    "5new_fin": "#C7DB15",
    "6die_fin":  "#7C880F",
    "7new_die_fin": "#282C04"
    }

    # HScurves["statCol"] = HScurves.status.map(color_map)

    return HScurves, color_map


def animate(SIM, save_path=None):

    dfData, color_map = preproc_curves_data(SIM, dopad = True)

    stats = dfData['stat'].unique().astype(str).tolist()
    stats.sort()

    #R upper bound
    Rub = SIM.hr*1.1
    
    dfData = dfData.sort_values(by='SimStep', ascending=False)

    fig = px.scatter(dfData, x="X", y="Y",
                        color="status", 
                        # animation_group="I",
                        range_x=[-Rub,Rub], range_y=[-Rub,Rub],
                        hover_data=["I", "SimStep"],
                        animation_frame="SimStep", 
                        animation_group="I",
                        color_discrete_map=color_map,
                        #name = "Front",
                        # direction= "counterclockwise", start_angle=0,
                        #color_discrete_sequence=px.colors.sequential.Plasma_r, 
                        #template="plotly_dark",)
                        
                        category_orders={"stat": stats},
                        width=700, height=600,
                        # render_mode="SVG"
                        )

    fig.update_layout(shapes = [casing(SIM.hr)])
 
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


def casing(R):
    (x0, y0, x1, y1) = np.array((-1,-1,1,1)) * R
    
    shape = dict(
            type="circle",
            x0=x0, y0=y0, x1=x1, y1=y1,
            xref="x", yref="y",            # Locks the circle to data coordinates
            line=dict(
                color="grey", 
                width=1,
                # dash="dash"                # Options: 'solid', 'dash', 'dot', 'dashdot'
            ),
            fillcolor="rgba(0, 0, 0, 0)"   # Keeps the inside transparent so data shows through
        )
    
    return shape


def WebAndPerf(curvesDF, simDF, hull_radius, save_path = None):

    #R upper bound
    Rub = np.ceil(hull_radius*1.01) 

    
    grain = px.line(curvesDF, x="X", y="Y",
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

    perf = px.line(simDF, x = 'SimStep', y = 'C', render_mode="SVG",)

    fig = make_subplots(subplot_titles=('Web burned', 'Performance (Steps vs. Circumference)' ), rows=1, cols=2)

    for trace in grain.data:
        fig.add_trace(trace, row=1, col=1)

    # fig.add_trace(hull_trace, row=1, col=1)

    fig.update_layout(shapes=[casing(hull_radius)])

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


def Shape(R, T):
    Rmax = np.max(R) 
    X, Y = pol2cart(R, T)
    mode='lines'
    fig = px.line(x=X, y=Y)
    fig.update_layout(width=500 , height = 500, 
                      yaxis_range = [-Rmax, Rmax], 
                      xaxis_range = [-Rmax, Rmax])
    fig.show()