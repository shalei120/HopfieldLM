#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m energy -b 64  -d enwiki8 -g 0 -layer 6 -emb 512 -nhead 4 -c BET-nadm -s dgx #> energyLM-wiki2.out
