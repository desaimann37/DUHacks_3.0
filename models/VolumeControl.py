from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui

app = Flask(__name__)

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


@app.route('/')
def index():
    return render_template('./templates/index.html')


@app.route('/vol_control')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
