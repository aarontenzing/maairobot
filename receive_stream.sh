#!/bin/bash

gst-launch-1.0 -v udpsrc port=5000 buffer-size=400000 caps="application/x-rtp,encoding-name=H265,payload=96" ! \
    rtpjitterbuffer latency=200 ! rtph265depay ! avdec_h265 ! videoscale ! videoconvert ! ximagesink

