#!/usr/bin/env bash


python mmdetection/tools/test.py \
mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
faster-pth/MSCOCO2014_epoch_12.pth \
--syn_weights Temp/66.pth \
--dataset coco --gzsd \
--a1 0.33 --a2 0.67 \
--out Temp/66g.pkl \
--iou 0.5

#python mmdetection/tools/test.py \
#mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#faster-pth/MSCOCO2014_epoch_12.pth \
#--syn_weights F_BU_Train/U5_3.0/r2/48.pth \
#--dataset coco --gzsd \
#--a1 0.33 --a2 0.67 \
#--out F_BU_Test/U5_3.0/r2/g48.pkl \
#--iou 0.5
#
#python mmdetection/tools/test.py \
#mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#faster-pth/MSCOCO2014_epoch_12.pth \
#--syn_weights Fasttext_BS_Train/B.65/r1/48.pth \
#--dataset coco --gzsd \
#--a1 0.33 --a2 0.67 \
#--out Fasttext_BS_Test/B.65/r1/g48.pkl \
#--iou 0.5

#a=0.75
#while (($(bc <<< "$a >=0.65")))
#do
#  for((i=45;i<=52;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#    faster-pth/MSCOCO2014_epoch_12.pth \
#    --syn_weights F_BUC_Train/U3.0C$a/r1/$i.pth \
#    --dataset coco --zsd \
#    --a1 0.33 --a2 0.67 \
#    --out F_BUC_Test/U3.0C$a/r1/$i.pkl \
#    --iou 0.5
#  done
#
#  for((i=45;i<=52;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#    faster-pth/MSCOCO2014_epoch_12.pth \
#    --syn_weights F_BUC_Train/U3.0C$a/r2/$i.pth \
#    --dataset coco --zsd \
#    --a1 0.33 --a2 0.67 \
#    --out F_BUC_Test/U3.0C$a/r2/$i.pkl \
#    --iou 0.5
#  done
#
#  a=$(bc <<< "$a-0.05")
#done


#for((i=45;i<=52;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#  faster-pth/MSCOCO2014_epoch_12.pth \
#  --syn_weights F_BU_Train/U5_1.0/r2/$i.pth \
#  --dataset coco --zsd \
#  --a1 0.33 --a2 0.67 \
#  --out F_BU_Test/U5_1.0/r2/$i.pkl \
#  --iou 0.5
#done


#a=2.6
#while (($(bc <<< "$a <=3.0")))
#do
#
#  for((i=45;i<=51;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#    faster-pth/MSCOCO2014_epoch_12.pth \
#    --syn_weights F_BU_Train/U5_$a/r1/$i.pth \
#    --dataset coco --zsd \
#    --a1 0.33 --a2 0.67 \
#    --out F_BU_Test/U5_$a/r1/$i.pkl \
#    --iou 0.5
#  done
#
#  for((i=45;i<=51;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#    faster-pth/MSCOCO2014_epoch_12.pth \
#    --syn_weights F_BU_Train/U5_$a/r2/$i.pth \
#    --dataset coco --zsd \
#    --a1 0.33 --a2 0.67 \
#    --out F_BU_Test/U5_$a/r2/$i.pkl \
#    --iou 0.5
#  done
#
#  a=$(bc <<< "$a+0.1")
#done
#


#a=0.45
#while (($(bc <<< "$a >=0")))
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#  faster-pth/MSCOCO2014_epoch_12.pth \
#  --syn_weights Fasttext_BS_Train/B$a/r1/48.pth \
#  --dataset coco --zsd \
#  --a1 0.33 --a2 0.67 \
#  --out Fasttext_BS_Test/B$a/r1/48.pkl \
#  --iou 0.5
#
#
#  python mmdetection/tools/test.py \
#  mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#  faster-pth/MSCOCO2014_epoch_12.pth \
#  --syn_weights Fasttext_BS_Train/B$a/r1/52.pth \
#  --dataset coco --zsd \
#  --a1 0.33 --a2 0.67 \
#  --out Fasttext_BS_Test/B$a/r1/52.pkl \
#  --iou 0.5
#
#  python mmdetection/tools/test.py \
#  mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#  faster-pth/MSCOCO2014_epoch_12.pth \
#  --syn_weights Fasttext_BS_Train/B$a/r2/48.pth \
#  --dataset coco --zsd \
#  --a1 0.33 --a2 0.67 \
#  --out Fasttext_BS_Test/B$a/r2/48.pkl \
#  --iou 0.5
#
#
#  python mmdetection/tools/test.py \
#  mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#  faster-pth/MSCOCO2014_epoch_12.pth \
#  --syn_weights Fasttext_BS_Train/B$a/r2/52.pth \
#  --dataset coco --zsd \
#  --a1 0.33 --a2 0.67 \
#  --out Fasttext_BS_Test/B$a/r2/52.pkl \
#  --iou 0.5
#
#  a=$(bc <<< "$a-0.05")
#done
