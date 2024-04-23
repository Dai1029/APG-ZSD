import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv import imread
from mmdet.apis import init_detector, inference_detector
from mmdet.core import bbox2roi
from mmdet.datasets.pipelines import Compose
from numpy import array
from torchvision.transforms import Resize, ToTensor

configs_file = "mmdetection/configs/faster_rcnn_r101_fpn_1x_voc_test.py"
# checkpoint = "faster-pth/MSCOCO2014_epoch_12.pth"
checkpoint = "faster-pth/PASCALVOC_epoch_4.pth"
model = init_detector(configs_file, checkpoint, device="cuda:0")

feature_1024 = torch.FloatTensor(300*4, 1024)

car = np.load("Diffusion_Feature/voc/car.npy")
dog = np.load("Diffusion_Feature/voc/dog.npy")
sofa = np.load("Diffusion_Feature/voc/sofa.npy")
train = np.load("Diffusion_Feature/voc/train.npy")
car = car.reshape((300, 512, 512, 3))
dog = dog.reshape((300, 512, 512, 3))
sofa = sofa.reshape((300, 512, 512, 3))
train = train.reshape((300, 512, 512, 3))

data = np.concatenate((car, dog), axis=0)
data = np.concatenate((data, sofa), axis=0)
data = np.concatenate((data, train), axis=0)
# data = data.reshape((300*15, 512, 512, 3))

np.save("Diffusion_Feature/voc_512.npy", data)

for i in range(data.shape[0]):
    img = data[i].astype(np.uint8)
    img = img.reshape((512, 512, 3))
    result = inference_detector(model, img)
    feat = model.bbox_head.feature_1024
    feature_1024.narrow(0, i, 1).copy_(feat.data.cpu())
np.save("Diffusion_Feature/voc_1024.npy", feature_1024)





