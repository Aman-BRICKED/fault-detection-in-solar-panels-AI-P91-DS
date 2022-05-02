import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('fault_detector.hdf5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # SOLAR PANEL FAULT DETECTOR
         """
         )

file = st.file_uploader("Please upload the Solar Panel image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np


def import_and_predict(image_data,model):
    
    size=(300,300 )
    image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload a Solar Panel Image")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    CATEGORIES = ['FAULTY', 'NOT FAULTY']
    string="This Solar Panel is most likely to be "+CATEGORIES[np.argmax(predictions)]
    st.success(string)
    
