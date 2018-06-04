import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15,
    'gpu': 1.0
}
tfnet = TFNet(options)

#capture = cv2.VideoCapture('http://166.254.127.186/mjpg/video.mjpg?resolution=704x480')
capture = cv2.VideoCapture('trafficVideo.mp4')
#http://166.254.127.186/view/view.shtml?id=764&imagepath=%2Fmjpg%2Fvideo.mjpg&size=2
#http://166.254.127.186/mjpg/video.mjpg?resolution=704x480
#http://166.254.127.185 worst quality
#colors = [tuple(255 * np.random.rand(3)) for i in range(50)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)

    if ret:
        counter = 0;
        for result in results:
            label = result['label']
            if label == 'car':
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                frame = cv2.rectangle(frame, tl, br, (0,0,255), 3)
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                counter = counter + 1

        cv2.imshow('frame', frame)
        print('Car Counter: ', counter)
        #print('FPS {:.1f}'.format(1/(time.time() - stime)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        capture.release()
        cv2.destroyAllWindows()
        break
