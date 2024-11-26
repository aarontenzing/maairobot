import serial
import time

ser = serial.Serial('/dev/ttyLEDS', 115200, timeout=1)
time.sleep(2)
ser.write('NOMOW/'.encode())
time.sleep(5)
ser.write('MOW/'.encode())
time.sleep(2)
ser.write('STOP/'.encode())
ser.close()
