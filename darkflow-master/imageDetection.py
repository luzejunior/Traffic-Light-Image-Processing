import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

#%config InlineBackend.figure_format = 'svg'

options = {
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolov2.weights',
    'threshold' : 0.3,
    'gpu' : 1.0
}

tfnet = TFNet(options)
img = cv2.imread('car.jpg', cv2.IMREAD_COLOR);
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)

for index in range(len(result)):
    tl = (result[index]['topleft']['x'], result[index]['topleft']['y'])
    br = (result[index]['bottomright']['x'], result[index]['bottomright']['y'])
    label = result[index]['label']
    img = cv2.rectangle(img, tl, br, (0,255,0), 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,255,255), 2)

plt.imshow(img)
plt.show()
