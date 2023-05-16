from io import BytesIO
from flask import Flask, request, render_template
from PIL import Image
from ultralytics import YOLO
import cv2
from utils import plot_results

model=YOLO("best.pt")

model.predict('./selam.jpg')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imageUploadSuccess')
def imageUploadSuccess():
    return render_template('index.html',  message = "Prediction result saved with name prediction.jpg")

@app.route('/imageUploader', methods=['POST'])
def upload():
    for i in range(0,100):
        # check request.files[key] exits
        if 'file'+str(i) in request.files:
            file = request.files['file'+str(i)]
            
            # Convert the file data to a Pillow Image object
            img = Image.open(BytesIO(file.read()))

            results = model.predict(source=img,save=True,conf=0.5,iou=0.5)

            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'

            labels = {0: u'__background__', 1: u'A', 2: u'B',3: u'C',4: u'D',5: u'E',6: u'F',7: u'G',8: u'H'
                    ,9: u'I', 10: u'J', 11: u'K', 12: u'L', 13: u'M', 14: u'N', 15: u'O', 16: u'P', 17: u'Q',
                    18: u'R', 19: u'S', 20: u'T', 21: u'U', 22: u'V', 23: u'W', 24: u'X', 25: u'Y', 26: u'Z'}

            result_image_data = plot_results(results, folder_path='./', image_data=img, labels=labels, result_name = '.jpg', save_image=False, return_image=True)

            im = Image.fromarray(result_image_data)
            im.save('prediction.jpg')     

    # Return a response indicating success
    return 'Image uploaded and processed successfully!'

if __name__ == '__main__':
    app.run()
