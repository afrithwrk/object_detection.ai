import streamlit as st 
import ultralytics
from PIL import Image
import tempfile

st.set_page_config(page_title='Object Detection',page_icon='🔎',layout='wide')

st.title('Object Detection using :blue[yolo] models')

st.subheader(f'''This page detect the objects present in the given image using YOLO model - Trained on the coco dataset ''')

input_image = st.file_uploader('Upload the image here ',type=['png','jpg','jpeg'])

if input_image:
    img = Image.open(input_image)
    st.image(img,caption='Uploaded Image')
    
    with tempfile.NamedTemporaryFile(suffix='.jpg',delete=False) as temp:
        img.save(temp)
        temp_path = temp.name
        
    model = ultralytics.YOLO('yolov3u.pt')
    
    result = model(temp_path)
    
    result_image = result[0].plot()
    st.image(result_image, caption='Detected Image')
    
    st.json(result[0].to_json())
