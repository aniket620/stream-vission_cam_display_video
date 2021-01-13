import cv2
from flask import Flask , Response , render_template


import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import warnings
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


graph_def = tf.compat.v1.GraphDef()
filename = "model.pb"
with tf.compat.v2.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


sess = tf.compat.v1.Session()

input_1 = sess.graph.get_tensor_by_name('image_tensor:0')

output_1 = sess.graph.get_tensor_by_name('detected_scores:0')
output_2 = sess.graph.get_tensor_by_name('detected_classes:0')
output_3 = sess.graph.get_tensor_by_name('detected_boxes:0')


label_dict = {0: "Biological_Element",
             1: "Blockage",
             2:"Crack"}

font = cv2.FONT_HERSHEY_SIMPLEX
i=1


# cam = cv2.VideoCapture('Test_Img_Vid.mp4')

cam = cv2.VideoCapture(0)
# result = cv2.VideoWriter('Predicted.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'),10,None) 



app = Flask(__name__, static_folder='static')

def stream():
    # while 1 :
    while(cam.isOpened()):  
        # __,frame = cam.read()
        __,x = cam.read()
        x = cv2.resize(x, (320,320))
        ans_1, ans_2, ans_3 = sess.run([output_1,output_2, output_3], feed_dict={input_1 : [x]})
        x1,y1,x2,y2 = ans_3[np.argmax(ans_1)]*320
        lx,ly = x1,y1
        if x1 < 50 :
            lx = 50
        if y1 < 100 : 
            ly = 100 
        if x1 > 300 :
            lx = 300
        if y1 > 300 :
            ly = 300
        bbox = cv2.rectangle(x, (x1, y1), (x2, y2), (255,0,0), 2)
     
        label = label_dict[int(ans_2[np.argmax(ans_1)])]
        cv2.putText(bbox,str(label), (lx,ly), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        # i=i+1

        imgencode = cv2.imencode('.jpg',bbox)[1]
       # imgencode = cv2.imencode('.jpg',frame)[1]

        strinData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+strinData+b'\r\n')
    cam.release()

@app.route('/video')
def video():
    return Response(stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def main():
    return render_template('index.html')





if __name__ == "__main__":
    # app.run()
    app.run(host='127.0.0.1', port=5002)
