#importing required libraries
import tensorflow
import io
import os
import numpy as np
import keras
from PIL import Image , ImageOps
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import streamlit as st
from werkzeug.utils import secure_filename
import h5py
import cv2
import base64

#loading model

model = load_model('covid_detector.h5')


#defining classes 
classes = ['covid_infected','Normal']

def predict(image1): 
    image = image1
    size=(224,224)

    image = ImageOps.fit(image,size)
    image = image.convert('RGB')

    image = img_to_array(image)
    # reshape data for the model

    image = image.reshape((1, image.shape[0], image.shape[1], 3))

    # model prediction
    result = model.predict(image)

    ## get label with max accuracy
    if result>0.5:
      label = 1
    else:
      label=0

    label_name = classes[label]

    return label 

 #main page
def main():
    main_bg = "covid19.jpg"
    main_bg_ext = "jpg"


    st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True)

    html_tempti ="""
    <div style="padding:5px">
    <h1 style="color:white;text-align:center;font-weight:bold;font-style:verdana;">Covid19 Detection using Chest X-ray</h1>
    </div>
    """
    st.markdown(html_tempti,unsafe_allow_html=True)


    
    html_temp="""
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">We are here to help U &#129309;</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding',False)
    
    uploaded_file = st.file_uploader("",type=['jpg','png','jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button("Predict"):
        if uploaded_file is None:
            raise Exception("image not uploaded, please refresh page and upload the image")
        st.write("")
        st.write("Classifying...")
        label = predict(image)
        st.write('Predicted lables is ',classes[label])

        html_warn="""
            <div style="padding:5px">
            <h1 style="color:red;text-align:center;font-weight:bold;font-style:verdana;">It seems like problem, consult Doctor </h1>
            </div>"""
        html_nowarn="""

            <div style="padding:5px">
            <h1 style="color:white;text-align:center;font-weight:bold;font-style:verdana;">You are alright,keep smile</h1>
            </div>"""

        

        if label==0:
            st.markdown(html_warn,unsafe_allow_html=True)
        else:

            st.markdown(html_nowarn,unsafe_allow_html=True)
            
    hide_streamlit_style ="""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    html_temp3="""
    <p style="color:white">This application can able to detect whether a person is infected with covid19 or not.</p>
    <p style="color:white"> Our Team
    <ul style="color:white">
    <li> M.poojitha</li>
    <li> N.Ganesh</li>
    <li> Pushpa</li>
    <li> Jyotsna</li>
    </ul></p>
    """
    
    if st.button("About"):
        st.markdown(html_temp3,unsafe_allow_html=True)

if __name__=='__main__':
    main()
    
