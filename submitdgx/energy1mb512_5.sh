#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m energy -b 64  -d 1mb -g 0 -s dgx -emb 512 -layer 5 #> energyLM-wiki2.out
