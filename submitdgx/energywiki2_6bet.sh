#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m energy -b 64  -d wiki2 -g 0 -layer 6 -emb 300 -nhead 4 -c BET -s dgx #> energyLM-wiki2.out
