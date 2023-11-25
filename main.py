import io
import cv2
from fastapi import FastAPI, UploadFile

# STEP1 : Load Packages
import easyocr
from fastapi.responses import StreamingResponse
import numpy as np

# STEP2 : Load Inference Module 
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

app = FastAPI()

@app.post('/predict')
async def predict_api(image_file: UploadFile):   
    # STEP3 : Load Input 
    # 0. read bytes from http
    content = await image_file.read()
    
    # # 1. make buffer from bytes
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # STEP4 : Inference
    result = reader.readtext(img, output_format='json')

    # STEP5 : Visualize Reult
    print(result)
    return{result}

@app.post("/predict/img")
async def predict_api_img(image_file: UploadFile):
    # 0. read bytes from http
    contents = await image_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # STEP 4 : Inference
    result = reader.readtext(img)
    # annotated_image = visualize(image_copy, result)
    for detection in result:
      #get min max coordinations
      x_min, y_min = [int(cord) for cord in detection[0][0]]
      x_max, y_max = [int(cord) for cord in detection[0][2]]
      #get text
      text = detection[1]
      # declare the font
      font = cv2.FONT_HERSHEY_SIMPLEX
      # draw rectangles
      img = cv2.rectangle(img, (x_min,y_min),(x_max,y_max),(0,255,0),2)
      # put the texts
      img = cv2.putText(img, text, (x_min, y_min),font, 1, (255, 25, 200),1, cv2.LINE_AA)
    # rgb_annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # STEP 6: encode
    img_encode = cv2.imencode('.png', img)[1]
    image_stream = io.BytesIO(img_encode.tobytes())
    return StreamingResponse(image_stream, media_type="image/png")