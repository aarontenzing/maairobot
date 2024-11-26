import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

class Stream(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/video_stream":
            self.send_response(200)
            self.send_header('content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            # Capture opencv webcam
            while True:
                ret, frame = cam.read()
                if not ret:
                    break

                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n\r\n")
                    #print("test")
                
                #time.sleep(0.02) # 20 FPS max

def start_server():
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, Stream)
    print("Server started at http://10.42.0.228:8080/video_stream")
    httpd.serve_forever()

cam = cv2.VideoCapture(9)
width = 640
height = 480
cam.set(3, width)
cam.set(4, height)

server_thread = threading.Thread(target=start_server())
server_thread.daemon = True
server_thread.start()

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow('camera', frame)
    cv2.waitKey(30)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()

