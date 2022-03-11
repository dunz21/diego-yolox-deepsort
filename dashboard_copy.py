
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
from flask_cloudflared import  run_with_cloudflared

import plotly.io as pio

dark = True
# Use only for dark Themes rest acan stay the same 
if dark:
    pio.templates.default = "plotly_dark"




# Init Flask Server
server = Flask(__name__)
run_with_cloudflared(server)
# Init Dash App
app = Dash(__name__, server = server, external_stylesheets=[dbc.themes.VAPOR,'https://fonts.googleapis.com/css2?family=Montserrat'])

# Init Tracker
tracker = Tracker(filter_classes= None, model = 'yolox-s', ckpt='weights/yolox_s.pth')

Main = []

# Sunburst Data Function
def build_hierarchical_dataframe(df, levels, value_column):
    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value'])
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i+1]].copy()
        else:
            df_tree['parent'] = 'total'
        df_tree['value'] = dfg[value_column]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='total', parent='',
                              value=df[value_column].sum(),
                              ))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees


def update_layout(figure, margin, Title):
    figure.update_layout(font_family = "Montserrat", 
        title=Title,
        xaxis={
            'autorange': True,
            'showgrid': False,
            'zeroline' :False,
            'automargin':True},
        margin=margin,
        yaxis={
            'autorange': True,
            'showgrid':False,
            'zeroline': False,
            'automargin': True,
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
            )
    return figure

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
# Card Compnent

def create_card(Header, Value, cardcolor):
    card = dbc.Col([
        dbc.Card([
            dbc.CardHeader(Header, style = {'text-align':'center'}),
            dbc.CardBody([
                html.H3(Value, style = {'text-align':'center'})
            ])
        ], color = cardcolor, inverse=True, style = {
            "width":"18rem",
            'text-align':"center",
            "vertical-align":"middle"
        })
    ])
    return card


# Video Feed Component
videofeeds = dbc.Col(style = {'padding':"0px 0px 0px 10px", 'padding-top' : '60px' }, width=4, children =[
        html.Img(src = "/video_feed", style = {
            'max-width':'100%',
            'height':'auto',
            'display':'block',
            'margin-left':'auto',
            'margin-right':'auto'})]) 

# Header Component

header = dbc.Col( width = 10, children=[
                html.Header(style = {
                    'padding': '10px 10px 10px 10px;',
                    'text-align': 'center;',
                    'background': '#1abc9c;',
                    'color': 'white;',
                            },children = [html.H1("Traffic Flow Management System", style = { 
                                'text-align': 'center', 
                                'font-size': '4.5rem',
                                'font-family':"Montserrat"})
                            ]),
             ])
# Grpahical Components
figure1 = dbc.Col([dcc.Graph(id="live-graph1")], width=4)
figure2 = dbc.Col(dcc.Graph(id="live-graph2"), width=4)
piefig = dbc.Col(dcc.Graph(id="piefig"), width=4)
dirfig = dbc.Col(dcc.Graph(id="dirfig"), width=4)
sunfig = dbc.Col(dcc.Graph(id="sunfig"), width=4)
speedfig = dbc.Col(dcc.Graph(id="speedfig"), width=8)
infig = dbc.Col(dcc.Graph(id="infig"), width=4)



fps = 0
res = "A x B"
stream = "Stream 1"
average_speed = 0
previous_av_speed = 0


"""
This Function Takes the input as n_interval and will execute by itself after a certain time
It outputs the figures 

"""
@app.callback([
    Output('live-graph1', 'figure'),
    Output('live-graph2', 'figure'),
    Output('cards', 'children'),
    Output('piefig', 'figure'),
    Output('dirfig', 'figure'),
    Output('sunfig', 'figure'),
    Output('speedfig', 'figure'),
    Output('infig', 'figure'),

    ],
        [
            Input('visual-update', 'n_intervals')
        ]   
)
def update_visuals(n):
    global average_speed, previous_av_speed
    fig1     = go.FigureWidget()
    fig2     = go.FigureWidget()
    piefig   = go.FigureWidget()
    dirfig   = go.FigureWidget()
    sunfig   = go.FigureWidget()
    speedfig = go.FigureWidget()
    infig    = go.FigureWidget()


    # Dataset Creation a
    vehicleslastminute = 0
    vehiclestotal = 0
    df = pd.DataFrame(Main)

    if len(df) !=0:        
        df1 = df.copy()
        df1['count'] = 1
        average_speed = int(df["Speed"].mean())
        # Database Transformations
        df = df.pivot_table(index = ['Time'], columns = 'Category', aggfunc = {'Category':"count"}).fillna(0)
        df.columns = df.columns.droplevel(0)
        df = df.reset_index()
        df.Time = pd.to_datetime(df.Time)
        columns = list(df.columns)
        columns.remove('Time')

        # Direction Datset
        dirdf = df1.groupby(['direction']).agg({"Speed": np.mean}).reset_index()

        # Sunburst Dataset
        df_all_trees = build_hierarchical_dataframe(df=df1, levels = ["Category",'direction'], value_column = "count")
     
        # Speed Dataset
        df1 = df1.pivot_table(index = ['Time'], columns = 'Category', aggfunc = {'Speed':np.mean}).fillna(0)
        df1.columns = df1.columns.droplevel(0)
        df1 = df1.reset_index()
        df1.Time = pd.to_datetime(df1.Time)
        columns1 = list(df1.columns)
        columns1.remove('Time')

        # Speed Fig Add Scatter 
        for col in columns1:
            speedfig.add_scatter(name = col, x = df1['Time'], y = df1[col], fill = "tonexty", line_shape = "spline")

        
        # Looping for adding scatter for each category
        values_sum = []
        for col in columns:    
            fig1.add_scatter(name = col,x=df['Time'], y=df[col], fill='tonexty', showlegend=True, line_shape='spline')
            fig2.add_scatter(name = col,x=df['Time'], y=df[col].cumsum(), fill='tonexty', showlegend=True, line_shape='spline')
            vehicleslastminute += df[col].values[-1]
            vehiclestotal += df[col].cumsum().values[-1]
            values_sum.append(df[col].sum())
        
        piefig = px.pie(
            labels = columns, names = columns, values = values_sum, hole = 0.5,
            title = "Traffic Distribution - Vehicle Type",
            color_discrete_sequence= px.colors.sequential.Agsunset, opacity=0.85 
        )

        dirfig = px.bar(dirdf, y = "direction", x = "Speed", color = "direction", orientation="h",hover_name="direction",
                color_discrete_map={
                "North" : "rgba(188,75,128,0.8)",
                "South" : 'rgba(26,150,65,0.5)',
                "East"  : 'rgba(64,167,216,0.8)',
                "West"  : "rgba(218,165,32,0.8)"},
            title = "Average Speed Direction Flow"

        )

        sunfig = go.FigureWidget(go.Sunburst(
            labels = df_all_trees['id'],
            parents = df_all_trees['parent'],
            values = df_all_trees['value'],
            branchvalues = 'total',
            textinfo = 'label+percent entry',
            opacity = 0.85
        ))

    cards = [
        create_card(Header = "Vehicles This Minute", Value = vehicleslastminute, cardcolor = "primary"),
        create_card(Header = "Total Vehicles", Value = vehiclestotal, cardcolor = "info"),

        create_card(Header = "Frames Per Second", Value = fps, cardcolor = "secondary"),
        create_card(Header = "Resolution", Value = res, cardcolor = "warning"),
        create_card(Header = "Stream", Value = stream, cardcolor = "danger"),


    ]
    
    infig = go.FigureWidget(
        go.Indicator(
            domain = {'x':[0,1], 'y':[0,1]},
            value = average_speed,
            mode = "gauge+number+delta",
            title = {'text':"Average Speed Km/h"},
            delta = {'reference': previous_av_speed},
            gauge = {'axis': {'range': [None, 50]},
             'bar': {'color': "rgba(50,251,226,60)"},

                'steps' : [
                    {'range': [0, 15], 'color': 'rgba(0,0,0,0)'},
                    {'range': [15, 50], 'color':'rgba(0,0,0,0)'}],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 45}}

        )
    )


    #Updating the layout
    
    fig1        = update_layout(figure=fig1, margin=dict(pad=20), Title='Traffic per Minute')
    fig2        = update_layout(figure=fig2, margin=dict(pad=20), Title='Cumulative Traffic')
    speedfig    = update_layout(figure=speedfig, margin=dict(pad=20), Title='Average Speed Flow by Vehicle Type')
    dirfig      = update_layout(figure=dirfig, margin=dict(t=40, b=10, r=10, l=10), Title="Average Speed Direction Flow")
    sunfig      = update_layout(figure=sunfig, margin=dict(t=30, b=10, r=60, l=10), Title="Traffic Direction Flow")
    infig       = update_layout(figure=infig, margin=dict(t=40, b=10, r=10, l=10), Title="Average Speed Km/h")
    piefig      = update_layout(figure=piefig, margin=dict(t=30, b=10, r=60, l=10), Title="Traffic Distribution - Vehicle Type")
    
    piefig.update_traces(textposition='inside', textinfo='percent+label')

    return fig1, fig2 , cards, piefig, dirfig, sunfig, speedfig, infig



app.layout = html.Div([
    # Input for all the updating visuals
    dcc.Interval(id='visual-update',interval=2000,n_intervals = 0),

    dbc.Row([header] ,style = {'padding-bottom':"20px", "padding-top":"20px"}), #Header
    dbc.Row(id="cards", style = {'padding-bottom':"20px", "padding-top":"20px"}), #Cards
    dbc.Row([videofeeds, figure1, figure2]), #VideoFeed and 2 Graphs
    dbc.Row([piefig, sunfig ,dirfig]), #Header
    dbc.Row([speedfig, infig]), #Header

])


if __name__ == '__main__':
    app.run_server(debug =True, port = 8050)