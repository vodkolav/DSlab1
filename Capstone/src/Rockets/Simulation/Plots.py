import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from Rockets.Geometry import pol2cart, cart2pol
from Rockets.utils import trypop, ia

import json
import ipywidgets as widgets
from IPython.display import display, Javascript

def dots_and_arrows(dfC, color_map, dfI ):   

    # dfC, color_map = preproc_curves_data(SIM, dopad = False, **filtr)

    dfC["statCol"] = dfC.status.map(color_map)


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

    # fig.update_layout(shapes = [casing(SIM.hr)])

    fig.update_layout(width=800, height=800)

    return fig


def Interactive_polar(df, **kwargs):
    # Interactive Polar plot
    #R upper bound
    Rub = np.ceil(df.R.max()) + 1

    fig = px.line_polar(df, r="R", theta="TD", line_close=True,
                        range_r=[0,Rub], animation_frame="Step", 
                        direction= "counterclockwise", start_angle=0,
                        #color_discrete_sequence=px.colors.sequential.Plasma_r, 
                        #template="plotly_dark",)
                        **kwargs
                        )
    fig.show()


def minipad(df, index = "SimStep", col = 'I', value = "stat"):
    """Pad a DataFrame to ensure that all combinations of index and columns are present.

    Args:
        df (pd.DataFrame): The input DataFrame to pad.
        index (str, optional): The name of the index column. Defaults to "SimStep".
        cols (list, optional): List of column names to consider for padding. Defaults to ['I'].
        value (str, optional): The name of the value column. Defaults to "stat".

    Returns:
        pd.DataFrame: The padded DataFrame.
    """

    # index = "SimStep"  

    # cols = ['I']

    # value = "stat" 

    lasti = df.groupby(index)[col].max()+1 # or min -1 ? 

    lasti = lasti.reset_index()

    Is = lasti.values 
    Ss = df[value].unique()

    Ss = np.expand_dims(Ss, axis = 1)
    Ss

    pld = [np.concat([a, b]) for a in Is for b in Ss]

    pld = pd.DataFrame(pld, columns=[ index, col, value])

    paddf = pd.concat([df,pld], axis=0)
    paddf = paddf.sort_values(by=["SimStep", "I","stat"]).reset_index(drop=True)
    return paddf


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


def preproc_log_data(logdata: dict, epoch_precision = "s"):
    colmap = { "ping": "green", "info": "blue", "warning": "orange", "error": "red"}
    ymap = {"ping": 0, "info": 1, "warning": 2, "error": 3}

    dflog = pd.DataFrame(logdata)
    dflog['y'] = dflog['type'].map(ymap) + 0.4 * (dflog.case_signature == "experiment")

    if epoch_precision == "s":
        mult = 1000
    elif epoch_precision == "ms":
        mult = 1
    else:
        raise ValueError("epoch_precision possible values: 's', 'ns'")
    dflog["time"] = dflog["time"]*mult
    dflog["dt"] = pd.to_datetime(dflog["time"], unit='ms')# must be in ms
    dflog['symbol'] = dflog.case_signature.apply(lambda l: 101 if l != "experiment" else 102)
    # dflog['level'] = 101 if 
    return dflog, colmap


def preproc_sim_data(HSsim):

    dfSim = pd.DataFrame(HSsim).dropna()
    dfSim.SimStep = dfSim.SimStep.apply(lambda x: x[0])
    dfSim.C = dfSim.C.apply(lambda x: x[0])
    return dfSim


def preproc_curves_data(HScurves, dopad = False, **kwargs):

    HScurves = pd.DataFrame(HScurves)

    HScurves["stat"] = HScurves.IsNew * 1  + HScurves.filt*2 + (~HScurves.A) * 4 

    if dopad:
        HScurves = minipad(HScurves,"SimStep", "I", "stat" )

    sm10 = (HScurves['SimStep']%10)

    HScurves['StepMod10'] = sm10.astype(str)

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

    HScurves["statCol"] = HScurves.status.map(color_map)

    HScurves['hovertext'] =  HScurves.apply( lambda r : "<br>".join([ f"I: {r.I}", f"St: {r.status}"]), axis=1)

    return HScurves, color_map


def preproc_intersections_data(HSintersections, **kwargs):

    dfI = HSintersections.results()
    
    dfI = pd.DataFrame(dfI)

    return dfI


def animate(curvesDF, hull_radius, color_map, **kwargs):

    # curvesDF, curve_colors = preproc_curves_data(tracks['log']['data'], dopad = True)

    stats = curvesDF['status'].unique().astype(str).tolist()
    stats.sort()

    #R upper bound
    Rub = hull_radius*1.1

    asc = trypop(kwargs, 'ascending', True)
    curvesDF = curvesDF.sort_values(by='SimStep', ascending=asc)

    lockscale = trypop(kwargs,"lockscale", False)

    fig = px.scatter(curvesDF, x="X", y="Y",
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
                        
                        category_orders={"status": stats},
                        **kwargs,
                        # render_mode="SVG"
                        )

    fig.update_layout(shapes = [casing(hull_radius)])
    fig.update_traces(marker=dict(size=4))

    if lockscale:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


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


def Web(curvesDF, hull_radius, **kwargs):

    #R upper bound
    Rub = np.ceil(hull_radius*1.01) 

    
    grain = px.line(curvesDF, x="X", y="Y",
                        # color="stat", 
                        range_x=[-Rub,Rub], range_y=[-Rub,Rub],
                        hover_data=["I", "SimStep"],
                        color='StepMod10',
                        # direction= "counterclockwise", start_angle=0,
                        #color_discrete_sequence=px.colors.sequential.Plasma_r, 
                        #template="plotly_dark",)
                        #category_orders={"stat": ["0", "1", "2", "3"]},
                        # width=700, height=600,
                        **kwargs
                        )
    grain.update_layout(shapes=[casing(hull_radius)])

    return grain


def Perf(simDF , **kwargs):

    perf = px.line(simDF, x = 'SimStep', y = 'C', **kwargs)
    return perf


def Log(dflog, colmap, **kwargs):

        more = trypop(kwargs, 'hover_data', [])

        hover_data = ['case_signature', 'case_index'] + ia(more)

        fig = px.scatter(dflog, x = 'time', y = 'y', color = 'type', #title = "Log data",
                        hover_data = hover_data, 
                        custom_data= [list(dflog.index)],
                        # labels = {'value': 'Value', 'time': 'Time (s)', 'type': 'Type'},
                        color_discrete_map = colmap, **kwargs )
        # fig.update_xaxes(type="date", tickformat="%H:%M %b %d %Y")
        # fig.update_traces(marker=dict(size=6,
        #                               symbol=dflog['symbol'],
        #                   line=dict(width=2,
        #                   color='DarkSlateGrey')))
        # fig.layout.yaxis.fixedrange = True
        return fig


def multiplot(rows=1, cols=2, width=600, height=600):
    
    width = width * cols
    height = height * rows
    
    fig = make_subplots( rows, cols, horizontal_spacing=0.1, vertical_spacing=0.1,
                        subplot_titles=('Web burned', 'Performance (Steps vs. Circumference)' ))

    # fig.update_layout(yaxis_range = [-Rmax, Rmax], xaxis_range = [-Rmax, Rmax], **self.conf("layout"))
    
    # 5) Optional layout tweaks
    fig.update_layout(width=width, height=height) #, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    fig.update_traces(marker=dict(size=1))
    return fig


def save_fig(fig, save_path = None):
    if save_path is not None:
        
        fig.write_html(save_path, auto_play = False )

    else:
        fig.show()


def add_plot(multiplt, plot, row, col):
    for trace in plot.data:
        multiplt.add_trace(trace, row=row, col=col)

    for f in plot.select_shapes():
        multiplt.add_shape(f, row=row, col=col)
    
    return multiplt



def Shape(**kwargs):
    if 'R' in kwargs and 'T' in kwargs:
        X, Y = pol2cart(R, T)

    elif 'X' in kwargs and 'Y' in kwargs:
        X = kwargs['X']
        Y = kwargs['Y']
        R,T = cart2pol(X,Y)
    else:
        raise ValueError("Shape function requires either 'R' and 'T' or 'X' and 'Y' keyword arguments.")

    Rmax = np.max(R) 
    mode='lines'
    fig = px.line(x=X, y=Y, render_mode='SVG')
    fig.update_layout(width=500 , height = 500, 
                      yaxis_range = [-Rmax, Rmax], 
                      xaxis_range = [-Rmax, Rmax])
    fig.show()


# Use a clean dataframe and keep only the fields you want
# source_df = logdf.copy()#.reset_index( names='eventId')

# Build a visible row payload for every point
# fields = ["time", "y", "type", "message", "case_signature", "case_index", 'eventId']
# payload = source_df[fields].copy()

# Make the plot with customdata attached to each point
# fig = px.scatter(
#     source_df,
#     x="time",
#     y="y",
#     color="type",
#     hover_data=["message", "case_signature", "case_index"],
#     custom_data= [list(source_df.index)],
#     title="Log data",
#     labels={"value": "Value", "time": "Time (s)", "type": "Type"},
#     color_discrete_map=log_cmap,
#     render_mode='SVG'
# )

def make_selector(fig, source_df):
    # fig.layout.yaxis.fixedrange = True
    fig.update_xaxes(type="date", tickformat="%H:%M \n %b %d %Y")
    fig_w = go.FigureWidget(fig,
                            layout=widgets.Layout(width="70%", 
                                                  height="500px",
                                                #   display="flex",
                                                #   flex_flow="row"
                                                  ))

    # fig_w.update_xaxes(type="date", tickformat="%H:%M %b %d %Y")

    copy_btn = widgets.Button(
        description="Copy to clipboard",
                layout=widgets.Layout(
                    width="90%",
                    display="flex",
                    flex_flow="col")
    )

    selected_box = widgets.Textarea(
        value="",
        placeholder="Click a point on the plot...",
        layout=widgets.Layout(
                    width="90%",
                    height="100%",
                    display="flex",
                    flex_flow="col"),
        white_space="pre",
        overflow="auto"
        # description="Selected point:",
        # description_display='initial'
    )

    def on_point_click(trace, points, selector):
        if len(points.point_inds) == 0:
            return
        # display(trace, points, selector)
        # This is the real clicked point payload from the trace
        pid = points.point_inds[0],
        pt = trace['customdata'][pid]
        evId = pt[0]

        row = dict(source_df.loc[evId,:])
        selected_box.value = json.dumps(row, default=str, indent=2)


    for tr in fig_w.data:
        tr.on_click(on_point_click)

    def copy_click(_):
        val = selected_box.value
        if not val:
            return
        display(Javascript(f"""
            navigator.clipboard.writeText({json.dumps(val)});
        """))

    copy_btn.on_click(copy_click)

    return widgets.HBox([
                fig_w,
                widgets.VBox([copy_btn, selected_box ],
                            layout=widgets.Layout(
                                width="30%",))],
                layout=widgets.Layout(
                    width="1500px",
                    display="flex",
                    flex_flow="row"))

