# Copyright (C) 2019 Eugene a.k.a. Realizator, stereopi.com, virt2real team
#
# This file is part of StereoPi tutorial scripts.
#
# StereoPi tutorial is free software: you can redistribute it 
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# StereoPi tutorial is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoPi tutorial.  
# If not, see <http://www.gnu.org/licenses/>.
#
#          <><><> SPECIAL THANKS: <><><>
#
# Thanks to Adrian and http://pyimagesearch.com, as a lot of
# code in this tutorial was taken from his lessons.
#  
# Thanks to RPi-tankbot project: https://github.com/Kheiden/RPi-tankbot
#
# Thanks to rakali project: https://github.com/sthysel/rakali


from picamera import PiCamera
from stereo_camera import *
import time
import cv2
import numpy as np
import json
from datetime import datetime

print ("You can press Q to quit this script!")
time.sleep (5)

# Camera settimgs
cam_width = 1280
cam_height = 480

# Final image capture settings
scale_ratio = 0.5

# Display the disparity
display_disparity = True
disp_max = -100000
disp_min = 100000

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Used camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
camera.hflip = True

# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("Disparity")
cv2.moveWindow("Disparity", 450,100)

calibration_data = np.load('calibration_data.npz')
        
left_map_1 = calibration_data['left_map_1']
left_map_2 = calibration_data['left_map_2']
right_map_1 = calibration_data['right_map_1']
right_map_2 = calibration_data['right_map_2']
Q = calibration_data['Q']
print(Q)

with open('sbm_config.json') as sbm_config_file:
    sbm_config = json.load(sbm_config_file)
    
sbm = create_SBM(sbm_config)

# capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    t1 = datetime.now()
    _3dImage, disparity, image_left, _ = compute_3dImage(sbm, frame, left_map_1, left_map_2, right_map_1, right_map_2, Q)


    image_left = cv2.cvtColor(image_left,cv2.COLOR_GRAY2RGB)
    image_left = cv2.rectangle(image_left, (155,115), (165,125), (0, 255, 0), thickness=2)
    
    #distance = _3dImage[120,160,2]
    distance = 0
    distance_map = _3dImage[115:125, 155:165, 2]
    distance_map[distance_map < 0] = 0.0
    n_valid_distance = np.count_nonzero(distance_map)
    if (n_valid_distance != 0):
        distance = np.sum(distance_map)/n_valid_distance

    cv2.putText(image_left,
                str(distance),
                (100,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)
    cv2.imshow("Image", image_left)
    
    if (display_disparity):                
        disp_max = max(disparity.max(),disp_max)
        disp_min = min(disparity.min(),disp_min)
        local_max = disp_max
        local_min = disp_min
        disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
        disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
        disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
        disparity_color = cv2.rectangle(disparity_color, (155,115), (165,125), (0, 255, 0), thickness=2)
        cv2.imshow("Disparity", disparity_color)

    t2 = datetime.now()
    #print ("DM build time: " + str(t2-t1))
    
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();



