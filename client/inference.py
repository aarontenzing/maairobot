import time
import argparse
import cv2
import numpy as np
import threading
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
from periphery import GPIO
from rknnlite.api import RKNNLite
import multiprocessing
import serial
import multiprocessing as mp


# Parse arguments
parser = argparse.ArgumentParser(description='Run model on rknn')
parser.add_argument("model")
args = parser.parse_args()

# Constants
url = "http://10.42.0.1:5000/upload"  # URL for POST requests to webserver
class_names = ["flower", "grass"]
camera_idx = 9

# Init Ledstrip
import time
import board
import neopixel_spi as neopixel
from led_strip import mow_animation, leds_white

print("--> Creating subprocess")
q = mp.Queue()
p = mp.Process(target=mow_animation, args=(q,))
p.start()
q.put("stop")

# Init RKNN
rknn_lite = RKNNLite()
print("--> Load RKNN model")
ret = rknn_lite.load_rknn(args.model)
if ret != 0:
    print("Load RKNN model failed")
    exit(ret)

print("--> Init runtime")
ret = rknn_lite.init_runtime()
if ret != 0:
    print("Init runtime failed")
    exit(ret)
print("Done")

# Camera
print("--> Preparing camera")
cam = cv2.VideoCapture(camera_idx)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution to reduce processing load
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cam.set(cv2.CAP_PROP_FPS, 10)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Stream Handler
class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/video_stream":
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            frame_skip = 10   # Process every 10 th frame 3 times inference per second 
            frame_count = 0
            post_delay = 5  # Delay between POST requests in seconds
            start = time.time() # Start time
            stop_mowing = time.time()
            class_name = ""
            #flower_score = 0 


            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                # Encode the frame as JPEG for streaming
                stream_image = frame.copy()
                cv2.putText(stream_image, f'Prediction: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                _, jpeg = cv2.imencode('.jpg', frame)
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                encoded_frame = jpeg.tobytes()
                self.wfile.write(encoded_frame)
                self.wfile.write(b"\r\n\r\n")
                
                # Skip frames to reduce processing load
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                print("--> Inference...") 
                # Prepare for inference
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = image[150:,:]
                image = cv2.resize(image, (224, 224))

                # Perform inference
                outputs = rknn_lite.inference(inputs=[image])
                print(outputs)
                idx = np.argmax(outputs[0][0])
                pred_score = softmax(outputs[0][0])[idx]
                class_name = class_names[idx]
                #print("Predicition: ", class_name)
                #print("outputs softmax: ", softmax(outputs[0][0]))

                #if class_name == "flower":
                #    flower_score += 1
                #else:
                #    flower_score = 0

                # Control the mowing LEDs -> when flower found stop mowing for time
                if time.time() - stop_mowing > 4:
                    if class_name == "flower" and pred_score > 0.7:
                        #mow_off()
                        q.put("off")
                        stop_mowing = time.time()
                    else:
                        q.put("on")
                        #mow_on()

                if (time.time() - start) > post_delay:
                    # Send POST
                    threading.Thread(target=self.send_post_request, args=(encoded_frame, idx)).start()
                    start = time.time()

    def send_post_request(self, encoded_frame, class_idx):
        # Prepare POST request data
        files = {"frame": ("frame", encoded_frame)}
        data = {"class": class_idx}
        try:
            response = requests.post(url, data=data, files=files)
            print("POST request sent:", response.status_code)
        except Exception as e:
            print("Failed to send POST request:", e)

def start_server():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, StreamHandler)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        ipaddr = s.getsockname()[0]
    print("Server started at http://" + ipaddr + ":8080/video_stream")
    httpd.serve_forever()

try:
    print("--> Starting server")
    start_server()

except KeyboardInterrupt:
    print("Server stopped by user")
    leds_white()

finally:
    cam.release()
