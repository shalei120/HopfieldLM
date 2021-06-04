#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m transformer -b 128  -d 1mb -g 0 -s dgx #> energyLM-wiki2.out
