#!/usr/bin/env python

"""
Run this file with:
GRADIO_SERVER_PORT=<port> python W13-L25-PalmOrPineGradioApp.py </path/to/model>

Where
- <port> is your server port number
- </path/to/model> is the path to the model you want to use for inference
"""

from fastai.vision.all import *
import gradio as gr


# Load the trained model
path = Path(sys.argv[1])
model = load_learner(path)


def classify(img):

    prediction = model.predict(img)
    label = prediction[0]
    label_index = prediction[1]
    probabilities = prediction[2]
    label_prob = probabilities[label_index]

    return f"I am {label_prob*100:.1f}% certain you've submitted a {prediction[0]} tree!"

title = "Palm Or Pine? I'll Decide!"
website = "A demo for [CS 152](https://cs.pomona.edu/classes/cs152/)" 

iface = gr.Interface(fn=classify, inputs=gr.inputs.Image(), outputs="text", title=title, article=website, theme="dark").launch()
