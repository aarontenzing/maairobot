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
from led_strip import mow_on, mow_off

NUM_PIXELS = 60
MOW_LEDS = 10
PIXEL_ORDER = neopixel.GRB
RED = 0xFF0000
GREEN = 0x00FF00
BLUE = 0x0000FF
DELAY = 0.005

# Initialize SPI and NeoPixel
print("--> Initialize SPI and NeoPixel")
spi = board.SPI()
pixels = neopixel.NeoPixel_SPI(
    spi, NUM_PIXELS, pixel_order=PIXEL_ORDER, brightness=0.5, auto_write=False
)

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
cam.set(cv2.CAP_PROP_FPS, 10)

# Stream Handler
class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/video_stream":
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            frame_skip = 30   # Process every th frame
            frame_count = 0
            post_delay = 10  # Delay between POST requests in seconds
            start = time.time() # Start time

            while True:
                ret, frame = cam.read()
                if not ret:
                    break

                # Encode the frame as JPEG for streaming
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
                class_name = class_names[idx]
                print("Predicition: ", class_name)

                # Control the mowing LEDs
                if class_name == "flower":
                    mow_off()
                else:
                    mow_on()

                #cv2.putText(frame, f'Prediction: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
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
    #start_server()
    frame = cv2.imread("dummy_data/validation/grass/0.jpg")
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = image[150:,:]
    image = cv2.resize(image, (224, 224))
    outputs = rknn_lite.inference(inputs=[image])
    print(outputs)
    idx = np.argmax(outputs[0][0])
    class_name = class_names[idx]
    print("Predicition: ", class_name)

except KeyboardInterrupt:
    print("Server stopped by user")

finally:
    cam.release()
