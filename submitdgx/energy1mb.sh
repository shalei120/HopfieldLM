#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m energy -b 128  -d 1mb -g 0 #> energyLM-wiki2.out