#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m transformer -b 64  -d enwiki8 -g 0 -layer 12 -emb 512 -nhead 8 -s dgx #> energyLM-wiki2.out
