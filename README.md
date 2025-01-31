## Project Overzicht

### Hardware
1. **Rock Pi 3A**: Runs a ResNet18 model for simple classification (flower, grass).
2. **Laptop**: 
   - Flask-webserver.
   - Embed images using a ResNet18 PlantNet model.
   - Dimensionality reduction (PCA) to visualize embeddings.
3. **Communication**:
   - WiFi hotspot on the laptop.
   - Rock Pi automatically connects to this hotspot.
4. **WS2812B Ledstrip**:
   - **Red**: Mowing disc disabled.
   - **Blue with white animation**: Mowing disc active.

## How to set up
1. **Set up the Flask Web Server (Host Machine)**:

Python version 3.12.3 

```bash
$ python3 -m venv .server

$ source .server/bin/activate

$ pip install -r requirements.txt

$ python3 webserver.py
```

2. **Set up the Client Grasrobot (ROCK PI)**:

```bash
$ python3 -m venv .grasrobot

$ source .grasrobot/bin/activate

$ pip install -r requirements.txt
```

  - On the client `ROCK Pi 3A`, install the [RKNN dependencies](https://github.com/airockchip/rknn-toolkit2). Follow this [guide](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RKNPU_Quick_Start_RKNN_SDK_V2.3.0_EN.pdf)!
  - In directory `client/`, you will find the code that runs on the `ROCK Pi 3A`.
  Run the following command:
  ```bash
  $ sudo python3 inference.py resnet18_flower_grass.rknn
  ```

3. **Set up WiFi communication between webserver and client**:

  - Create a WiFi hotspot on the laptop:  (SSID: `biobot`, Password: `biobot123biobot`)
  - The `ROCK Pi 3A` should automatically connect to this hotspot.
  - In the `inference.py` script on the client, change the web server IP to the laptop's IP. Then run the following command on the laptop:

  ```bash
  $ python3 webserver.py 
  ```
 
## Training model 
- The model is trained using PyTorch. It is a simple ResNet18 classifier trained on images of size 640x480. A horizontal crop of 640x330 is taken, and the images are resized to an input resolution of 224x224.

- Convert the PyTorch model to ONNX format first: [Tutorial](https://medium.com/@lahari.kethinedi/convert-custom-pytorch-model-to-onnx-9c7397366904)
- Finally, convert the ONNX model to RKNN format so that the model can run on the Rock Pi NPU. 
- Follow the **Rockchip RKNPU Quick Start Guide**: [Link to guide](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RKNPU_Quick_Start_RKNN_SDK_V2.3.0_EN.pdf).
- Use the **rknn-toolkit2** to install dependencies: [Toolkit repo](https://github.com/airockchip/rknn-toolkit2/).
- Refer to the **RKNN Model Zoo** for examples: [Model Zoo repo](https://github.com/airockchip/rknn_model_zoo).

Change the path to the ONNX file and RKNN output in conversion_script.py:
```python
DEFAULT_ONNX_PATH = '../model/imagenet_best_model.onnx'
DEFAULT_RKNN_PATH = '../model/imagenet_best_model.rknn'
```

In the RKNN Model Zoo directory, replace `resnet.py` with `conversion_script.py`.

## Dataset

The dataset for flower and grass classification can be found on the Apollo at `/avc/datasets/maairobot` or `/apollo/datasets/maairobot`.

## Key Challenges and Solutions

### WiFi Communication with Rock Pi
- **Problem**: TP-link dongles did not work out of the box (plug-and-play).
- **Solution**:
  - Manually install device drivers in the kernel.
  - Use a list of Linux-compatible WiFi adapters: [Morrownr USB WiFi repo](https://github.com/morrownr/USB-WiFi/blob/main/home/USB_WiFi_Adapters_that_are_supported_with_Linux_in-kernel_drivers.md).

### Controlling the WS2812B LED Strip via GPIO
- **Challenge**: Continuous animation loop blocks other processes. Controlling via GPIO pins on the Rock Pi.
- **Solution**: Implement multiprocessing:
  - Use the multiprocessing library to run parallel processes: [Documentation](https://docs.python.org/3/library/multiprocessing.html).
  - Create a Queue to send commands to the LED function: [Queue documentation](https://docs.python.org/3/library/queue.html#queue-objects).
- **Solution**: NeoPixel library on Rock Pi:
  - Controlling RGB LEDs only works via the SPI pin (pin 19). Refer to the [GPIO pin diagram](https://wiki.radxa.com/Rock3/hardware/3a/gpio). This pin must first be activated in the overlay file. Use [NeoPixels with rock](https://forum.radxa.com/t/how-to-use-neopixels-with-rock-pi-s/10492).

---

