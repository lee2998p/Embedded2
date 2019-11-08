#!/usr/bin/env python3
import torch
import video_encode
import cv2
import numpy as np
import struct
width = 852
height = 480
cap = cv2.VideoCapture(0)
writer = video_encode.VideoEncoder("test_vid", width, height, ROI=True)
count = 0
ROI = torch.zeros([1, 4], dtype=torch.int32)
while ";;":
    _, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    torch_frame = torch.from_numpy(frame)
    torch_frame = torch_frame.cuda()
    writer.write(torch_frame, ROI)
    cv2.imshow("FRAME", frame)
    count += 1
    if cv2.waitKey(1) == 27 or count == 400:
        break
print(f"Saved {count} frames")

print("OPENING file")
#reader = video_encode.VideoDecoder("test_vid.hevc")
#count = 0
#try:
#    while ";;":
#        t_frame = reader.read_frame()
#        count += 1
#except:
#    print(f"Read {count} frames")
