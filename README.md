# Landroid
## Project Outline

1. Set up the Flask Web Server: 
- python3 -m venv .myvenv
- srouce .myvenv/bin/activate
- pip install -r requirements.txt
- python3 webserver.py
2. Set up the client
- sudo python3 run.py resnet50_retrained_grass_flower.rknn

Example of POST request: curl -X POST http://127.0.0.1:5000/upload -F "frame=@/home/tenzing/Pictures/flower2.jpg" -F "class=1"