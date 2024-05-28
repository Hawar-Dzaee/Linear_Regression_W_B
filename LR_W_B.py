import streamlit as st 
import torch
import numpy as np 
import plotly.graph_objects as go


#----------Global Variables-----------


torch.manual_seed(42)

lower_bound = -1.
upper_bound =  1.
sample_size = 11
feature_1 = torch.linspace(lower_bound,upper_bound,sample_size)
inputs_for_line = torch.linspace(lower_bound-2,upper_bound+2,sample_size)
ground_truth = feature_1 + 0.2 


#----------Scatter & Line Plot [aka Figure 1]-----------

def generate_plot(w,b):

  Datapoints = go.Scatter(
  x = feature_1,
  y = ground_truth ,
  mode = 'markers',
  name = 'Data points'
)
  
  line = go.Scatter(
      x = inputs_for_line,
      y = w * inputs_for_line + b,
      mode = 'lines',
      name = 'model'
  )


  figure = go.Figure(data=[line, Datapoints])

  figure.update_layout(
    xaxis = dict(
      range = [-2,2],
      title = 'X',
      zeroline = True,
      zerolinewidth = 2,
      zerolinecolor = 'rgba(205, 200, 193, 0.7)'
    ),
    yaxis = dict(
      range = [-2,2],
      title = 'Y',
      zeroline = True,
      zerolinewidth = 2,
      zerolinecolor = 'rgba(205, 200, 193, 0.7)'
    ),
    height = 500,
    width = 2600
  )
  return figure

#----------Grid formation for the loss function----------
 

weight_combo = torch.linspace(-5,5,150)
bias_combo = torch.linspace(-5,5,150)

W,B = torch.meshgrid(weight_combo,bias_combo,indexing='ij')
W_F = W.flatten()
B_F = B.flatten()

L_F = []

for w,b in zip(W_F,B_F):
  l = torch.mean((ground_truth-( w * feature_1 + b))**2)
  L_F.append(l)


min_index = np.argmin(L_F)  # np.argmin does mind if the input is python list
np.unravel_index(min_index,W.shape)

L_F = torch.tensor(L_F)

#----------Loss Landscape------------------------------------

def loss_landscape(w,b):

  grid = go.Surface(
    x = W ,
    y = B,
    z = L_F.reshape(W.shape),
    colorscale = 'thermal',
    opacity = 0.3,
    name = 'Loss function landscape'
    
  )

  Global_minima = go.Scatter3d(
    x = (1,),
    y = (0.2,),
    z = (torch.min(L_F),),
    mode = 'markers',
    marker = dict(color='yellow',size=10,symbol='diamond'),
    name = 'Global minima'
  )

  ball = go.Scatter3d(
    x = (w,),
    y = (b,),
    z = (torch.mean((ground_truth-(w*feature_1 + b))**2),) ,
    mode = 'markers',
    marker = dict(color='red',size = 7),
    name = 'loss'
  )

  figure = go.Figure(data=[grid,Global_minima,ball])



                
  figure.update_layout(

    scene = dict
    (
    xaxis = dict(
      range = [-5,12],
      title = 'w',
      zeroline = True,
      zerolinewidth = 2,
      zerolinecolor = 'rgba(205, 200, 193, 0.7)'
    ),
    yaxis = dict(
      range = [-5,12],
      title = 'b',
      zeroline = True,
      zerolinewidth = 2,
      zerolinecolor = 'rgba(205, 200, 193, 0.7)'
    ),
    zaxis = dict(
      title = 'loss'
    ),
    camera =  dict(eye=dict(x=1.8, y=1.7, z=1.3)),
  ),

  legend=dict(
    x=1,  
    yanchor="top",
    y=0.99,
    xanchor="right"
  ),
  height = 500,
  width = 2600

  )
  return figure


#------------------------------------------------------------------------------------------------------------------------------------------------------
# streamlit 

st.set_page_config(layout='wide')


st.title("Linear Regression")
st.write('By : Hawar Dzaee')



with st.sidebar:
    st.subheader("Adjust the parameters to minimize the loss")
    w_val = st.slider("weight (w):", min_value=-4.0, max_value=4.0, step=0.1, value= -3.5)
    b_val = st.slider("bias   (b)", min_value=-4.0, max_value=4.0, step=0.1, value= -3.2)


container = st.container()

with container:
 
    st.write("")  # Add an empty line to create space

    # Create two columns with different widths
    col1, col2 = st.columns([3,3])

    # Plot figure_1 in the first column
    with col1:
        figure_1 = generate_plot(w_val, b_val)
        st.plotly_chart(figure_1, use_container_width=True, aspect_ratio=5.0)  # Change aspect ratio to 1.0
        st.latex(r'''\hat{y} = wX + b''')

    # Plot figure_2 in the second column
    with col2:
        figure_2 = loss_landscape(w_val, b_val)
        st.plotly_chart(figure_2, use_container_width=True, aspect_ratio=5.0)
        st.latex(r"""\text{MSE(w,b)} = \frac{1}{n} \sum_{i=1}^n (\ y_i- (wX + b) )^2""")


# -----------

# This is the after the latest 
