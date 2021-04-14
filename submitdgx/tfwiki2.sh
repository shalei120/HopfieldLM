#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m transformer -b 128  -d wiki2 -g 0 #> energyLM-wiki2.out
