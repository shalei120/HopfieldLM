#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m energy -b 128  -d wiki2 -g 0 -layer 3 -s dgx #> energyLM-wiki2.out
