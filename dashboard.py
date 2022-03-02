# Handle warnings
import warnings
warnings.filterwarnings("ignore")

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

from dash import Dash, html, dcc
from flask import Flask
from datetime import date

import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components  as dbc


# Exceptional Imports to avoid error on Local
from trackerMain import Tracker


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

tracker = Tracker(filter_classes=None, model='yolox-s', ckpt='weights/yolox_s.pth')    # instantiate Tracker


fps = 0
res = "Calculating..."
stream = "Stream 1"
average_speed = 0
previous_av_speed = 0

Main = deque(maxlen=100)


class VideoCamera(object):
    def __init__(self):
        global res;
        # self.video = cv2.VideoCapture('/Users/rohitkukreja/Downloads/2_2.mp4')
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
            try:
                image, bbox, data = tracker.update(image,logger_=False) 
                Main.extend(data)
            except:
                pass
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


server = Flask(__name__)


app = Dash(__name__, server = server, external_stylesheets= [dbc.themes.BOOTSTRAP])




@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div(style = {'padding': '5px 18px 20px 20px'},
    children=[
        # for header
        dbc.Row(style = {'padding': '30px 10px 40px 10px' ,'align-items':'center'}, children=[
            dbc.Col(width = 1, children=[html.Img(src ='assets/logo1.png', height = '80px')]),
            dbc.Col( width = 10, children=[
                html.Header(style = {
                    'padding': '10px 10px 10px 10px;',
                    'text-align': 'center;',
                    'background': '#1abc9c;',
                    'color': 'white;',
                    'font-size': '30px;',
                            },children = [html.H1("Traffic Flow Management System", style = { 'text-align': 'center'})
                            ]),
             ]),
         ]), 


        # for video and inputs
        dbc.Row([
            dbc.Col(width = 4, style = {'padding':"0px 0px 0px 10px", 'padding-top' : '60px' }, children =[
                html.Div(style = {
                    'width': '100%',
                    'background-color': 'white',
                    'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)',
                    },
                    children=[
                        html.Img(src="/video_feed", style= {
                        'max-width': '100%',
                        'height':'auto',
                        'display': 'block',
                        'margin-left': 'auto',
                        'margin-right': 'auto'        
                        })
                    ])
            ]),
            
        ]
    )])
    
if __name__ == '__main__':
    app.run_server(debug=True, port ='8050')



