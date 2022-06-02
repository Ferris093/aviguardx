'''
Run predict.py with saved model parameters:
1. Put the picture you want to predict in the project or change the path of _img_.(circulating.jpg)
2. Run predict.py with pytorch environment
    1. Define a model: Make sure nets and utils folders are in the project.
    2. Load trained parameters for the model: my_voc_weights_resnet.pth in model_data.
3. Output of the model
    1. predicted_img.jpg is the predicted picture with rectangle marks. Classes and scores(confidence) of recognized objects are labeled.
    2. List _coordinates_ represents the top, left, bottom and right coordinates of the recognized objects.

P.S.
1. Use os.listdir() to traverse the folder, and use Image.open to open the image file for prediction
2. If you want to intercept the target, you can use the obtained top, left, bottom, right to intercept the original image.
'''

from frcnn import FRCNN
from PIL import Image

img = Image.open('straightLine.jpg')
img.show()
# print(img)

frcnn = FRCNN()

image = img
image = image.convert("RGB")
r_image, prediction = frcnn.detect_image(image)
r_image.save("output.jpg")
if len(prediction)==0:
    print("No object")
else:
    print('left, top, right, bottom coordinates:')
    print(prediction)
    r_image.show()
