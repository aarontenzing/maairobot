import time
import board
import neopixel_spi as neopixel
import multiprocessing as mp

NUM_PIXELS = 82
MOW_LEDS = 20
PIXEL_ORDER = neopixel.GRB
RED = 0xFF0000
GREEN = 0x00FF00
BLUE = 0x0000FF
WHITE = 0xFFFFFF
DELAY = 0.0005

# Initialize SPI and NeoPixel
spi = board.SPI()
pixels = neopixel.NeoPixel_SPI(
    spi, NUM_PIXELS, pixel_order=PIXEL_ORDER, brightness=1, auto_write=False
)

def leds_white():
    pixels.fill(WHITE)
    pixels.show()

def leds_off():
    pixels.fill(0)
    pixels.show()

def mow_off():
    """ Turn all LEDs red. """
    pixels.fill(RED)
    pixels.show()

def mow_on():
    """ Turn all LEDs blue. """
    pixels.fill(BLUE)
    pixels.show()

def mow_animation(q):
    """ Turn LEDs green and animate blue moving lights for the specified duration. """
    color1 = WHITE
    color2 = WHITE
    while True:

        for i in range(NUM_PIXELS - MOW_LEDS):
            if not q.empty():
                item = q.get()
                if item == "on":
                    #print("on-sub")
                    color1 = WHITE
                    color2 = BLUE
                elif item == "off":
                    #print("off-sub")
                    color1 = RED
                    color2 = RED
                else:
                    print("stop-sub")
                    color1 = WHITE
                    color2 = WHITE

            # Set the moving LEDs to blue
            for j in range(MOW_LEDS):
                pixels[i+j] = color1
            # Display the updated pixel colors
            pixels.show()
            time.sleep(DELAY)
            pixels.fill(color2)

if __name__ == "__main__":
    q = mp.Queue()
    p = mp.Process(target=mow_animation, args=(q,))
    p.start()
    while True:
        q.put("on")
        print("on")
        time.sleep(5)
        q.put("off")
        print("off")
        time.sleep(5)
