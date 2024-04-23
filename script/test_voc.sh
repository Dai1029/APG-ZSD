#!/usr/bin/env bash

for((i=129;i<=129;i++));
do
  python mmdetection/tools/test.py \
  mmdetection/configs/VOC_1402.py \
  faster-pth/PASCALVOC_epoch_4.pth \
  --syn_weights VOC_BS/B.80/r3/$i.pth \
  --dataset voc --gzsd \
  --a1 0.35 --a2 0.65 \
  --out VOC_BS_test/B.80/r3/g$i.pkl \
  --iou 0.5
done


#for((i=129;i<=129;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/ORI/r3/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/ORI/r3/$i.pkl \
#  --iou 0.5
#done
#a=1.2
#while (($(bc <<< "$a <2.0")))
#do
#  a=$(bc <<< "$a+0.2")
#
#  for((i=113;i<=113;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BU/15_U$a/r1/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BU_test/15_U$a/r1/$i.pkl \
#    --iou 0.5
#  done
#
#done
#for((i=110;i<=136;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_BU/5_U.6/r1/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_BU_test/5_U.6/r1/$i.pkl \
#  --iou 0.5
#done

#
#for((i=122;i<=122;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/S_01/r2/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/S_01/r2/$i.pkl \
#  --iou 0.5
#done

#
#for((i=131;i<=132;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/S0/r2/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/S0/r2/$i.pkl \
#  --iou 0.5
#done
#
#for((i=131;i<=132;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/S_1/r1/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/S_1/r1/$i.pkl \
#  --iou 0.5
#done
#
#for((i=131;i<=132;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/S_1/r2/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/S_1/r2/$i.pkl \
#  --iou 0.5
#done
#
#for((i=131;i<=132;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/S_01/r1/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/S_01/r1/$i.pkl \
#  --iou 0.5
#done
#
#for((i=131;i<=132;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_Base/S_01/r2/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_Base_test/S_01/r2/$i.pkl \
#  --iou 0.5
#done


#a=0
#while (($(bc <<< "$a <0.3")))
#do
#  a=$(bc <<< "$a+0.2")
#
#  for((i=131;i<=131;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BU/New/5_U$a/r3/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BU_Test/New/5_U$a/r3/$i.pkl \
#    --iou 0.5
#  done
#
#done
#
#a=0.1
#while (($(bc <<< "$a <0.4")))
#do
#  a=$(bc <<< "$a+0.2")
#
#  for((i=131;i<=131;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BU/New/5_U$a/r1/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BU_Test/New/5_U$a/r1/$i.pkl \
#    --iou 0.5
#  done
#
#    for((i=131;i<=131;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BU/New/5_U$a/r2/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BU_Test/New/5_U$a/r2/$i.pkl \
#    --iou 0.5
#  done
#
#  for((i=131;i<=131;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BU/New/5_U$a/r3/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BU_Test/New/5_U$a/r3/$i.pkl \
#    --iou 0.5
#  done
#
#done
#
#a=0.90
#while (($(bc <<< "$a >=0.65")))
#do
#  for((i=129;i<=129;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BS/B$a/r1/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BS_Test/B$a/r1/$i.pkl \
#    --iou 0.5
#  done
#
#  for((i=129;i<=129;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BS/B$a/r2/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BS_Test/B$a/r2/$i.pkl \
#    --iou 0.5
#  done
#
#
#  for((i=129;i<=129;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BS/B$a/r3/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BS_Test/B$a/r3/$i.pkl \
#    --iou 0.5
#  done
#
#  for((i=129;i<=129;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BS/B$a/r4/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BS_Test/B$a/r4/$i.pkl \
#    --iou 0.5
#  done
#
#
#  for((i=129;i<=129;i++));
#  do
#    python mmdetection/tools/test.py \
#    mmdetection/configs/VOC_1402.py \
#    faster-pth/PASCALVOC_epoch_4.pth \
#    --syn_weights VOC_BS/B$a/r5/$i.pth \
#    --dataset voc --zsd \
#    --a1 0.35 --a2 0.65 \
#    --out VOC_BS_Test/B$a/r5/$i.pkl \
#    --iou 0.5
#  done
#
#  a=$(bc <<< "$a-0.05")
#done


#for((i=130;i<=135;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_BS/B.80/r4/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_BS_Test/B.80/r4/$i.pkl \
#  --iou 0.5
#done
#
#for((i=137;i<=140;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_BS/B.80/r4/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_BS_Test/B.80/r4/$i.pkl \
#  --iou 0.5
#done

#for((i=130;i<=135;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_BS/B.80/r5/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_BS_Test/B.80/r5/$i.pkl \
#  --iou 0.5
#done
#
#for((i=137;i<=140;i++));
#do
#  python mmdetection/tools/test.py \
#  mmdetection/configs/VOC_1402.py \
#  faster-pth/PASCALVOC_epoch_4.pth \
#  --syn_weights VOC_BS/B.80/r5/$i.pth \
#  --dataset voc --zsd \
#  --a1 0.35 --a2 0.65 \
#  --out VOC_BS_Test/B.80/r5/$i.pkl \
#  --iou 0.5
#done