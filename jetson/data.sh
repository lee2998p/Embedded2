#!/bin/bash

for i in $(seq 0 4); do
  echo "Processing $i"
  python3 face_extractor.py --cuda 1 --rate 5 --input_dir ~/data/$i --output_dir ~/faces/$i --trained_model ../models/ssd300_WIDER_100455.pth
done
