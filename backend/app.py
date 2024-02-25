from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Create a Flask web application
app = Flask(__name__)


# Testing Camera Capturing...
"""
camera = cv2.VideoCapture(0)
def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
"""


# 1. Virtual Mouse : 

"""
This file is for using Virtual Mouse using Opencv, Mediapipe & pyautogui 
We can run it over Flask server as we run this file 
To run this file : python VirtualMouse.py
"""

# app = Flask(__name__)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0  # Initialize index coordinates

def generate_mouse_frames():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    if id == 8:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                        index_x = screen_width / frame_width * x
                        index_y = screen_height / frame_height * y

                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    if id == 4:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                        thumb_x = screen_width / frame_width * x
                        thumb_y = screen_height / frame_height * y
                        print('outside', abs(index_y - thumb_y))
                        if abs(index_y - thumb_y) < 20:
                            pyautogui.click()
                            pyautogui.sleep(1)
                        elif abs(index_y - thumb_y) < 100:
                            pyautogui.moveTo(index_x, index_y)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 2. Virtual Painter : 


# Constants : 
# For proper indentation of video 
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

def getTool(x):
    if x < 50 + ml:
        return "line"

    elif x < 100 + ml:
        return "rectangle"

    elif x < 150 + ml:
        return "draw"

    elif x < 200 + ml:
        return "circle"

    else:
        return "erase"

def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True

    return False

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# drawing tools
tools = cv2.imread("./templates/tools.png")
tools = tools.astype('uint8')

mask = np.ones((480, 640)) * 255
mask = mask.astype('uint8')

def generate_painter_frames():
    global curr_tool  # Add this line

    cap = cv2.VideoCapture(0)
    ml = 150
    max_x, max_y = 250 + ml, 50
    curr_tool = "select tool"
    time_init = True
    rad = 40
    var_inits = False
    thick = 4
    prevx, prevy = 0, 0

    if not cap.isOpened():
        print("Error: Could not open camera.")

    while True:
        _, frm = cap.read()
        if frm is None:
            print("Error: Empty frame received.")
            continue
        frm = cv2.flip(frm, 1)

        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        op = hand_landmark.process(rgb)

        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
                x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

                if x < max_x and y < max_y and x > ml:
                    if time_init:
                        ctime = time.time()
                        time_init = False
                    ptime = time.time()

                    cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)
                    rad -= 1

                    if (ptime - ctime) > 0.8:
                        curr_tool = getTool(x)
                        print("your current tool set to : ", curr_tool)
                        time_init = True
                        rad = 40

                else:
                    time_init = True
                    rad = 40

                if curr_tool == "draw":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
                        prevx, prevy = x, y

                    else:
                        prevx = x
                        prevy = y

                elif curr_tool == "line":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not (var_inits):
                            xii, yii = x, y
                            var_inits = True

                        cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)

                    else:
                        if var_inits:
                            cv2.line(mask, (xii, yii), (x, y), 0, thick)
                            var_inits = False

                elif curr_tool == "rectangle":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not (var_inits):
                            xii, yii = x, y
                            var_inits = True

                        cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)

                    else:
                        if var_inits:
                            cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                            var_inits = False

                elif curr_tool == "circle":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        if not (var_inits):
                            xii, yii = x, y
                            var_inits = True

                        cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)

                    else:
                        if var_inits:
                            cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (0, 255, 0), thick)
                            var_inits = False

                elif curr_tool == "erase":
                    xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                    y9 = int(i.landmark[9].y * 480)

                    if index_raised(yi, y9):
                        cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)
                        cv2.circle(mask, (x, y), 30, 255, -1)

        op = cv2.bitwise_and(frm, frm, mask=mask)
        frm[:, :, 1] = op[:, :, 1]
        frm[:, :, 2] = op[:, :, 2]

        frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

        cv2.putText(frm, curr_tool, (270 + ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("paint app", frm)


        _, buffer = cv2.imencode('.jpg', frm)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 


# 3. Volume Controller : 
        
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

x1 = y1 = x2 = y2 = 0


def generate_frames():
    while True:
        _, image = cap.read()
        image = cv2.flip(image, 1)
        frame_height, frame_width, _ = image.shape

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = my_hands.process(rgb_image)
        hands = output.multi_hand_landmarks
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(image, hand)
                landmarks = hand.landmark
                for id, landmark in enumerate(landmarks):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    if id == 8:
                        cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                        x1 = x
                        y1 = y
                    if id == 4:
                        cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                        x2 = x
                        y2 = y
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5) // 4
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if dist > 50:
                pyautogui.press("volumeup")
            else:
                pyautogui.press("volumedown")

        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/virtual_mouse')
def video_feed1():
    return Response(generate_mouse_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/virtual_painter')
def video_feed2():
    return Response(generate_painter_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vol_control')
def video_feed3():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)













