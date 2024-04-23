from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import torch
from collections import OrderedDict
import pandas as pd
import numpy as np
import os
from mmdet.datasets import build_dataset
from mmdet.apis.runner import copy_synthesised_weights_inference
from mmcv import Config

# build the model from a config file and a checkpoint file
config_file = '/media/dxm/D/DXM/Model/ZSD-Classifier-Enhance/mmdetection/configs/faster_rcnn_r101_fpn_1x.py'
checkpoint_file = '/media/dxm/D/DXM/Model/ZSD-Classifier-Enhance/faster-pth/MSCOCO2014_epoch_12.pth'
# checkpoint_file = 'cp_48_17/epoch_12.pth'
# syn_weights = '/home/disk/DXM/Model/zero_shot_detection-master/checkpoint_train/fasttext/fasttext_3syn_pro/classifier_best.pth'  # base

syn_weights = '/media/dxm/D/DXM/Model/ZSD-Classifier-Enhance/Paper_Train/BUCE/r1/74.pth'

score_thr = 0.45
try:
    os.makedirs('/media/dxm/D/DXM/DG-Result/coco35_g')
except OSError:
    pass

# import pdb; pdb.set_trace()

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = Config.fromfile(config_file)
dataset = build_dataset(cfg.data.test, {'test_mode': True})

# copy_syn_weights(syn_weights, model)
seen_bg_weight, seen_bg_bias = copy_synthesised_weights_inference(model, syn_weights, 'coco', split='65_15')
model.bbox_head.seen_bg_weight = torch.from_numpy(seen_bg_weight).cuda()
model.bbox_head.seen_bg_bias = torch.from_numpy(seen_bg_bias).cuda()

root = '/media/dxm/D/DXM/dataset/COCO2014/val2014'
# df = pd.read_csv('../MSCOCO/validation_coco_unseen_all.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name}" for file_name in file_names]
# files_path = np.array(files_path)
# img_infos
# for idx, img in enumerate(files_path[:1000]):
# import pdb; pdb.set_trace()
import random
# color = "%06x" % random.randint(0, 0xFFFFFF)
from splits import COCO_ALL_CLASSES

color_map = {label: (random.randint(0, 255), random.randint(120, 255), random.randint(200, 255)) for label in
             COCO_ALL_CLASSES}

# det_results = mmcv.load('/home/disk/DXM/Model/zero_shot_detection-master/checkpoint_test/base/1.pkl')
# det_results = mmcv.load('/media/dxm/D/DXM/Model/ZSD-Classifier-Enhance/Paper_Test/BUCE/r1/74g.pkl')
img_infos = dataset.img_infos

zsd = [
    'COCO_val2014_000000008676.jpg', 'COCO_val2014_000000009527.jpg', 'COCO_val2014_000000029357.jpg',
    'COCO_val2014_000000040686.jpg', 'COCO_val2014_000000044261.jpg',
]
#
# for idx, info in enumerate(img_infos):
#     img = f"{root}/{info['filename']}"
#     if info['filename'] in zsd:
#         result = inference_detector(model, img)
#         # result = det_results[start+idx]#inference_detector(model, img)
#         # import pdb; pdb.set_trace()
#         out_file = f"/media/dxm/D/DXM/DG-Result/coco35/{img.split('/')[-1]}"
#         show_result(f"{img}", result, model.CLASSES, out_file=out_file, show=False, score_thr=score_thr)
#         print(f"[{idx:03}/{len(img_infos)}]")

for idx, info in enumerate(img_infos):
    img = f"{root}/{info['filename']}"
    result = inference_detector(model, img)
    out_file = f"/media/dxm/D/DXM/DG-Result/coco35_g/{img.split('/')[-1]}"
    show_result(f"{img}", result, model.CLASSES, out_file=out_file, show=False, score_thr=score_thr)
    print(f"[{idx:03}/{len(img_infos)}]")
