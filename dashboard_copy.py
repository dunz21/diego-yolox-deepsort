
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
            if len(data) >0:
                print(data)
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


"""
This Function Takes the input as n_interval and will execute by itself after a certain time
It outputs the figures 

"""
@app.callback([
    Output('live-graph1', 'figure'),
    Output('live-graph2', 'figure'),
    ],
        [
            Input('visual-update', 'n_intervals')
        ]   
)
def update_visuals(n):
    fig1     = go.FigureWidget()
    fig2     = go.FigureWidget()
    
    # Dataset Creation 
    df = pd.DataFrame(Main)
    print(len(df))


    # Database Transformations
    df = df.pivot_table(index = ['Time'], columns = 'Category', aggfunc = {'Category':"count"}).fillna(0)
    df.columns = df.columns.droplevel(0)
    df = df.reset_index()
    df.Time = pd.to_datetime(df.Time)
    columns = df.columns
    
    # Looping for adding scatter for each category
    for col in columns:    
        if col == "Time":
            continue
        fig1.add_scatter(name = col,x=df['Time'], y=df[col], fill='tonexty', showlegend=True, line_shape='spline')
        fig2.add_scatter(name = col,x=df['Time'], y=df[col].cumsum(), fill='tonexty', showlegend=True, line_shape='spline')
    
    return fig1, fig2 



app.layout = html.Div([
    # Input for all the updating visuals
    dcc.Interval(id='visual-update',interval=1000,n_intervals = 0),

    dbc.Row([header]), #Header
    dbc.Row([]), #Cards
    dbc.Row([videofeeds, figure1, figure2]), #VideoFeed and 2 Graphs


])


if __name__ == '__main__':
    app.run_server(debug =True, port = 8050)