




from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

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
    fig = make_subplots(rows=1, cols=2, subplot_titles=('R vs T', 'Y vs X'))

    fig.add_trace(go.Scatter(x=X, y=Y, mode=mode, name='Y(X)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=T, y=R, mode=mode, name='R(T)'), row=1, col=2)

    fig.update_yaxes(title_text='Y', row=1, col=1, scaleanchor = 'x', scaleratio=1)

    fig.update_layout(width=1000 , height = 600)
    return fig