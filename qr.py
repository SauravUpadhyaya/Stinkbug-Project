import qrcode
import streamlit as st
from PIL import Image
from IPython.display import display

url = 'https://stinkbug-detection.streamlit.app/'

qr_img = qrcode.make(url)
display(qr_img)
qr_img.save('streamlit_qr.png')
