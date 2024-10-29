#!/bin/bash

gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H265 ! rtph265depay ! avdec_h265 ! videoscale ! videoconvert ! ximagesink


