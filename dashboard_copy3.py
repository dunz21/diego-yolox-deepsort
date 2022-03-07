
# General Imports
import cv2
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import deque

# Flask Imports
from flask import Flask, Response

# Plotly-Dash Imports 

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px

from flask import Flask
import dash_bootstrap_components  as dbc

from mainTracker import Tracker, vis_track, draw_lines, lines


# Init Flask Server
server = Flask(__name__)

# Init Dash App
app = Dash(__name__, server = server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Init Tracker
tracker = Tracker(filter_classes= None, model = 'yolox-s', ckpt='weights/yolox_s.pth')

Main = []


# -------------------------------------------------Getting Video Feeds ------------------------------#

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class VideoCamera(object):
    def __init__(self):
        global res;
        self.video = cv2.VideoCapture(sys.argv[1])
        res = f"{int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))}" 

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        global fps;
        success, image = self.video.read()
        if success:
            t1 = time_synchronized()
            image = draw_lines(lines, image)
            image, bbox, data = tracker.update(image, logger_=False)
            image = vis_track(image, bbox)
            Main.extend(data)
            fps  = f"{int((1./(time_synchronized()-t1)))}"
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return "Video is Completed !!!"

def gen(camera):
    fps = 0.0
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------------------------------------------------------------------------------#
# Card Component
def create_card(Header, Value, cardcolor):
    card = dbc.Col([
        dbc.Card( [
            dbc.CardHeader(Header, style = {'text-align': 'center'}),
            dbc.CardBody([html.H3(Value, className="card-title", style = {'text-align': 'center'})])],
            color=cardcolor, inverse=True, style={
                "width": "18rem",
                'text-align': 'center',
                'vertical-align': 'middle'
                })
            ])
    return card


# Video Feed Component
videofeeds = dbc.Col(width=4, children =[
        html.Img(src = "/video_feed", style = {
            'max-width':'100%',
            'height':'auto',
            'display':'block',
            'margin-left':'auto',
            'margin-right':'auto'})]) 

# Header Component
header = dbc.Col(width = 10,
    children = [ html.H1("Traffic Flow Management", style = {'text-align':'center'})]
)

# Grpahical Components
figure1 = dbc.Col([dcc.Graph(id="live-graph1")], width=4)
figure2 = dbc.Col(dcc.Graph(id="live-graph2"), width=4)
piefig  = dbc.Col(dcc.Graph(id="pie-fig"), width=4)



fps = 0
res = "Calculating..."
stream = "Stream 1"


"""
This Function Takes the input as n_interval and will execute by itself after a certain time
It outputs the figures 

"""
@app.callback([
    Output('live-graph1', 'figure'),
    Output('live-graph2', 'figure'),
    Output('cards'      , 'children'),
    Output('pie-fig'    , 'figure'),
    ],

    [Input('visual-update', 'n_intervals')]   
)
def update_visuals(n):
    fig1 = go.FigureWidget()
    fig2 = go.FigureWidget()
    piefig = go.FigureWidget()
    
    # Dataset Creation 
    vehicleslastminute = 0
    vehiclestotal = 0
    df = pd.DataFrame(Main)
    if len(df) !=0:        
        # Database Transformations
        df = df.pivot_table(index = ['Time'], columns = 'Category', aggfunc = {'Category':"count"}).fillna(0)
        df.columns = df.columns.droplevel(0)
        df = df.reset_index()
        df.Time = pd.to_datetime(df.Time)
        columns = df.columns
        columns.remove('Time')
       
        # Looping for adding scatter for each category
        values_sum = []
        for col in columns:    
            fig1.add_scatter(name = col,x=df['Time'], y=df[col], fill='tonexty', showlegend=True, line_shape='spline')
            fig2.add_scatter(name = col,x=df['Time'], y=df[col].cumsum(), fill='tonexty', showlegend=True, line_shape='spline')
            vehicleslastminute += df[col].values[-1]
            vehiclestotal += df[col].cumsum().values[-1]
            values_sum.append(df[col].sum())
        
        # Pie Fig (has to be inside the IF statement)
 
        piefig = px.pie(
        labels=columns, names = columns, values=values_sum, hole=.5,
        title = "Traffic Distribution - Vehicle Type",
        color_discrete_sequence=px.colors.sequential.Agsunset, opacity=0.85)
        
    cards = [
        create_card(Header= "Vehicles This Minute", Value = vehicleslastminute, cardcolor = "primary"),
        create_card(Header = "Total Vehicles", Value = vehiclestotal ,cardcolor="info"),
        create_card(Header = "Frames Per Second", Value= fps ,cardcolor="primary"),
        create_card(Header = "Resolution", Value= res ,cardcolor="warning"),
        create_card(Header = "Video Stream", Value= stream,cardcolor="danger")
    ]

 
    return (fig1, fig2, cards, piefig )



app.layout = html.Div([
    # Input for all the updating visuals
    dcc.Interval(id='visual-update',interval=2000,n_intervals = 0),

    dbc.Row([header]), #Header
    dbc.Row(id="cards"), #Cards
    dbc.Row([videofeeds, figure1, figure2]), #VideoFeed and 2 Graphs
    dbc.Row([piefig]),#pie , sunburst dirfig

])


if __name__ == '__main__':
    app.run_server(debug =True, port = 8050)