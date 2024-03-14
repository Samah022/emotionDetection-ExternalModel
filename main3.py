# import requests
# import cv2
# import re
# import cv2
# import time
# import base64
# import asyncio
# import numpy as np

# url = "https://faceanalyzer-ai.p.rapidapi.com/faceanalysis"

# payload = { "url": "rtsp://192.168.253.114:8080/h264_ulaw.sdp" }
# headers = {
# 	"content-type": "application/x-www-form-urlencoded",
# 	"X-RapidAPI-Key": "SIGN-UP-FOR-KEY",
# 	"X-RapidAPI-Host": "faceanalyzer-ai.p.rapidapi.com"
# }


# cap = cv2.VideoCapture(0)
# frame_count = 0
# frame_array = []

# while True:
#     success, frame =  asyncio.get_event_loop().run_in_executor(None, cap.read)

#     if not success:
#         break
#     response = requests.post(url, data=payload, headers=headers)