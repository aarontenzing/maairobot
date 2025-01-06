## Project Overzicht

### Hardware
1. **Rock Pi 3A**: Draait een ResNet18-model voor eenvoudige classificatie (bloem, gras).
2. **Laptop**: 
   - Flask-webserver.
   - Embed afbeeldingen met behulp van een ResNet18 PlantNet-model.
   - Dimensiereductie (PCA) om embeddings visueel weer te geven.
3. **Communicatie**:
   - Wifi-hotspot op laptop
   - Rock Pi maakt automatisch verbinding met deze hotspot.
4. **WS2812B Ledstrip**:
   - **Rood**: Maaischijf uitgeschakeld.
   - **Blauw met witte animatie**: Maaischijf actief.

---
## How to set up
1. **Set up the Flask Web Server**: 
  - python3 -m venv .myvenv
  - source .myvenv/bin/activate
  - pip install -r requirements.txt
  - python3 webserver.py
2. **Set up the Client Grasrobot**
  -  python3 -m venv .myvenv
  - source .myvenv/bin/activate
  - pip install -r requirements.txt
  - Op de client `ROCK Pi 3A` installeer de [RKNN dependencies](https://github.com/airockchip/rknn-toolkit2) , best deze [guide](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RKNPU_Quick_Start_RKNN_SDK_V2.3.0_EN.pdf) volgen 
  - In de [repo](https://github.com/aarontenzing/Biobot-server) vind je een directory `client`, kopieer deze directory naar de `ROCK Pi 3A`
  - Run vervolgens: sudo python3 inference.py resnet18_flower_grass.rknn
  3. Set up WiFi communication webserver en client: 
   - Maak WiFi hotspot op laptop:  (SSID: `biobot`, Wachtwoord: `biobot123biobot`)
   - Laat `ROCK Pi 3A` hiermee verbinden:
   - Verander inference.py op client, het IP van de webserver naar het IP van de laptop
 
---
## Model en Bestanden
- Model op de Rock Pi: `imagenet_best_model.rknn`.
- Training van model via PyTorch, eenvoudig ResNet18, afbeelding van 640x480 steeds gecropt --> horizontale balk 640x330. 
- Input resolutie is 224x224 

---
## Belangrijke Uitdagingen en Oplossingen

### 1. Wifi-communicatie met Rock Pi
- **Probleem**: TP-link dongles werkten niet direct (plug-and-play).
- **Oplossing**:
  - Manuele installatie van device drivers in de kernel.
  - Gebruik van een lijst met Linux-compatibele wifi-adapters: [Morrownr USB WiFi repo](https://github.com/morrownr/USB-WiFi/blob/main/home/USB_WiFi_Adapters_that_are_supported_with_Linux_in-kernel_drivers.md).

### 2. PyTorch Model op Rockchip NPU
- **Converteer naar RKNN Formaat**:
  - Volg de **Rockchip RKNPU Quick Start Guide**: [Link naar handleiding](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/01_Rockchip_RKNPU_Quick_Start_RKNN_SDK_V2.3.0_EN.pdf).
  - Gebruik de **rknn-toolkit2** voor installatie van dependencies: [Toolkit repo](https://github.com/airockchip/rknn-toolkit2/).
  - Raadpleeg de **RKNN Model Zoo** voor voorbeelden: [Model Zoo repo](https://github.com/airockchip/rknn_model_zoo).
- **Modelconversieproces**:
  1. Converteer PyTorch-model naar ONNX: [Tutorial](https://medium.com/@lahari.kethinedi/convert-custom-pytorch-model-to-onnx-9c7397366904).
  2. Converteer ONNX-model naar RKNN-formaat (voorbeeld in RKNN model zoo resnet.py).

### 3. Aansturing van WS2812B Ledstrip via GPIO
- **Uitdaging**: Continue animatieloop blokkeert andere processen. Aansturen via GPIO pins Rock.
- **Oplossing**: Multiprocessing implementeren:
  - Gebruik de **multiprocessing** library om parallelle processen te draaien: [Documentatie](https://docs.python.org/3/library/multiprocessing.html).
  - Maak een **Queue** aan om commando’s naar de led-functie te sturen: [Queue documentatie](https://docs.python.org/3/library/queue.html#queue-objects).
- **Oplossing**: NeoPixel library op Rock PI:
  - Aansturen van RGB LEDs werkt enkel via SPI pin,  [Schema GPIO pins](https://wiki.radxa.com/Rock3/hardware/3a/gpio) dus pin 19.  Deze moet je eerst activeren in overlay file, use [NeoPixels with rock](https://forum.radxa.com/t/how-to-use-neopixels-with-rock-pi-s/10492)

---

