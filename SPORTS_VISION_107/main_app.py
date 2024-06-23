class_names = ['ultimate',
'bowling',
'basketball',
'olympic wrestling',
'tug of war',
'shot put',
'wheelchair racing',
'badminton',
'sumo wrestling',
'football',
'cricket',
'skydiving',
'figure skating women',
'table tennis',
'luge',
'snow boarding',
'rowing',
'disc golf',
'snowmobile racing',
'ice climbing',
'sky surfing',
'harness racing',
'shuffleboard',
'bmx',
'bobsled',
'canoe slamon',
'judo',
'jousting',
'lacrosse',
'steer wrestling',
'bike polo',
'ice yachting',
'wheelchair basketball',
'speed skating',
'track bicycle',
'hang gliding',
'field hockey',
'billiards',
'baseball',
'high jump',
'tennis',
'water polo',
'archery',
'horse jumping',
'croquet',
'ski jumping',
'fencing',
'hammer throw',
'cheerleading',
'weightlifting',
'swimming',
'horseshoe pitching',
'mushing',
'frisbee',
'formula 1 racing',
'nascar racing',
'wwe',
'rugby',
'bungee jumping',
'curling',
'uneven bars',
'gymnastics',
'hockey',
'giant slalom',
'golf',
'balance beam',
'rings',
'parallel bar',
'water cycling',
'kabaddi',
'ampute football',
'pole vault',
'javelin',
'hurdles',
'pole dancing',
'gaga',
'volleyball',
'figure skating pairs',
'boxing',
'motorcycle racing',
'chess',
'surfing',
'air hockey',
'wingsuit flying',
'polo',
'rock climbing',
'rollerblade racing',
'shooting',
'jai alai',
'chuckwagon racing',
'figure skating men',
'bull riding',
'baton twirling',
'pole climbing',
'hydroplane racing',
'horse racing',
'log rolling',
'fly fishing',
'sailboat racing',
'arm wrestling',
'sidecar racing',
'pommel horse',
'roller derby',
'axe throwing',
'trapeze',
'barell racing',
'american football']


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

from util import classify, set_background

set_background("background.jpg")

# set title
st.title('Sports classification')

# set header
st.header('Please upload a sport image')

#small guide
st.write("Please Note: This pretrained model is fine-tuned on 107 types of sports categories. This model can have few errors as it is still under development")

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = tf.keras.models.load_model('fine_tuned_efficientnetb0_model.h5')

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, confidence = classify(image, model, class_names)

    # write classification
    st.write("## Prediction: {}".format(class_name))
    st.write("### Confidence score: {}%".format(int(confidence * 1000) / 10))
