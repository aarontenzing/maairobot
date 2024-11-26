import cv2

cam = cv2.VideoCapture(9)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#framerate = cam.set(cv2.CAP_PROP_FPS, 20)

frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
framerate = cam.get(cv2.CAP_PROP_FPS)

print(f"Resolution: {frame_width} {frame_height}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, framerate, (int(frame_width),int(frame_height)))

try:
    print("Capturing...")
    while True:
        ret, frame = cam.read()
        out.write(frame)

except KeyboardInterrupt:
    print("Stopped capturing")

finally:
    print("Cleaning resources")
    cam.release()
    out.release()
