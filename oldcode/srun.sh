#!/bin/sh
srun --time=20:00:00 -o ~/stn/code/r.out -e ~/stn/code/r.err --gres=gpu:1 nvidia-docker run --rm -u $(id -u):$(id -g) --mount type=bind,source=/Midgard/home/lukasfi/.local/,target=/.local --mount type=bind,source=/Midgard/home/lukasfi/stn/,target=/stn nvcr.io/nvidia/tensorflow:18.08-py3 /bin/bash -c "cd /stn/code && python3 stn_learn.py" &
