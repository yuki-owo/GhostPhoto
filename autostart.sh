#!/bin/bash

# ディスプレイが利用可能になるまで待機
export DISPLAY=:0
cd /home/ghost/Ghost/

wait_for_display() {
    while true; do
        result=$(xset q 2>&1)
        if [[ $result != *"unable to open display"* ]]; then
            echo "Display is available"
            break
        else
            echo "Waiting for display..."
            sleep 2
        fi
    done
}

wait_for_display

# 仮想環境をアクティブにしてmain.pyを実行
source /home/ghost/Ghost/venv/bin/activate
python3 /home/ghost/Ghost/main.py


