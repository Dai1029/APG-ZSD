#!/usr/bin/env bash
#
#python tools/zero_shot_utils.py \
#mmdetection/configs/faster_rcnn_r101_fpn_1x_iou.py \
#--load_from work_dirs_faster/coco_e12/epoch_12.pth \
#--save_dir /media/hz/dataset/DXM/ExtractFeature/E12_1 \
#--data_split test --classes unseen

#python tools/zero_shot_utils.py \
#mmdetection/configs/faster_rcnn_r101_fpn_1x.py \
#--load_from faster-pth/MSCOCO2014_epoch_12.pth \
#--save_dir /home/disk/DXM/Feature/E12 \
#--data_split train --classes seen

#python mmdetection/tools/zero_shot_utils.py \
#mmdetection/configs/VOC_1402.py \
#--load_from faster-pth/PASCALVOC_epoch_4.pth \
#--save_dir /media/dxm/D/DXM/Feature/VOC_Feature \
#--data_split train --classes seen

python mmdetection/tools/zero_shot_utils.py \
mmdetection/configs/VOC_1402.py \
--load_from faster-pth/PASCALVOC_epoch_4.pth \
--save_dir /media/dxm/D/DXM/Feature/VOC_Feature \
--data_split test --classes unseen