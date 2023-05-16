from roboflow import Roboflow
import cv2

rf = Roboflow(api_key="3L6w38yF5vD69WZxaK2F")
project = rf.workspace().project("braille-detection-vxtp1")
model = project.version(1).model

# infer on a local image
#print(model.predict("braille1.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("selam.jpg", confidence=50, overlap=50).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
img = cv2.imread("prediction.jpg")
  
# Displaying the image
cv2.imshow('image', img)

cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()