from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

def detect_objects():
    # open the webcam
    cap = cv2.VideoCapture(0)

    # initialize the object detector
    object_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # process frames from the webcam
    while True:
        # read a frame from the video feed
        ret, frame = cap.read()

        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect objects using the object detector
        objects = object_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # draw rectangles around the detected objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # encode the frame as a JPEG image
        _, jpeg = cv2.imencode('.jpg', frame)

        # yield the frame as JPEG data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        # stop processing frames if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the resources
    cap.release()

net = cv2.dnn.readNet('config/yolov3.weights', 'config/yolov3.cfg')

classes = []
with open("config/coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0) # 0 for webcam, or you can use the index of your camera if you have multiple cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set the width of the frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set the height of the frame

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def gen_frames():
    while True:
        ret, img = cap.read()
        if not ret:
            break

        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes)>0.2:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/detect')
def detect():
    """Start object detection."""
    return render_template('result.html', message='Object detection started!')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)