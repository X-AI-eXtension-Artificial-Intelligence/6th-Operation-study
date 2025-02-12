#!/bin/bash

source /c/xai_6th/.venv/Scripts/activate

#LOGFILE="log/train_$(date +'%Y%m%d_%H%M').log"
#echo "Log file save at $LOGFILE"
#python train.py  | tee "$LOGFILE" # 이걸로 하면 출력이 안보여서
python -u train.py

#>:표준 출력을 파일로 리다이렉션