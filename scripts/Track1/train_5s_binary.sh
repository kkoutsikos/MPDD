#!/usr/bin/env bash
set -e

# Default Training Parameters
data_rootpath="D:/HACI/MMchallenge/NEUQdata" # 数据集根目录
AUDIOFEATURE_METHOD="opensmile" # 音频特征类别,可选{wav2vec,opensmile,mfccs}
VIDEOLFEATURE_METHOD="resnet" # 视频特征类别，可选{openface, resnet, densenet}
SPLITWINDOW="5s" # 窗口时长，可选{"1s","5s"}
LABELCOUNT=2 # 标签分类数，可选{2, 3, 5}
TRACK_OPTION="Track1"
FEATURE_MAX_LEN=5 # 设定最大特征长度，不足补零、超出截断
BATCH_SIZE=2
LR=0.000018
NUM_EPOCHS=200
DEVICE="cpu"


for arg in "$@"; do
  case $arg in
    --data_rootpath=*) data_rootpath="${arg#*=}" ;;
    --audiofeature_method=*) AUDIOFEATURE_METHOD="${arg#*=}" ;;
    --videofeature_method=*) VIDEOLFEATURE_METHOD="${arg#*=}" ;;
    --splitwindow_time=*) SPLITWINDOW="${arg#*=}" ;;
    --labelcount=*) LABELCOUNT="${arg#*=}" ;;
    --track_option=*) TRACK_OPTION="${arg#*=}" ;;
    --feature_max_len=*) FEATURE_MAX_LEN="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --lr=*) LR="${arg#*=}" ;;
    --num_epochs=*) NUM_EPOCHS="${arg#*=}" ;;
    --device=*) DEVICE="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

for i in `seq 1 1 1`; do
    cmd="python train.py \
        --data_rootpath=$data_rootpath \
        --audiofeature_method=$AUDIOFEATURE_METHOD \
        --videofeature_method=$VIDEOLFEATURE_METHOD \
        --splitwindow_time=$SPLITWINDOW \
        --labelcount=$LABELCOUNT \
        --track_option=$TRACK_OPTION \
        --feature_max_len=$FEATURE_MAX_LEN \
        --batch_size=$BATCH_SIZE \
        --lr=$LR \
        --num_epochs=$NUM_EPOCHS \
        --device=$DEVICE"

    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done