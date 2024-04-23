# from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls import TrainCls
from train_gan import TrainGAN
from generate import load_unseen_att, load_all_att
# from mmdetection.splits import get_unseen_class_labels

opt = parse_args()


try:
    os.makedirs(opt.outname)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

for arg in vars(opt):
    print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)

torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

unseen_attributes, unseen_att_labels = load_unseen_att(opt)
attributes, _ = load_all_att(opt)
# init classifier
trainCls = TrainCls(opt)

print('initializing GAN Trainer')


seenDataset = FeaturesGAN(opt)

real_feature_bg, real_label_bg = seenDataset.getBG_and_seen_feat(400, 200)
# concatenate synthesized + real bg features
syn_feature = np.load("Diffusion_Feature/all_unseen_1024.npy")
label = [5, 7, 13, 16, 22, 29, 30, 32, 43, 49, 53, 62, 65, 71, 79]
num = 300
syn_label = torch.LongTensor(15 * num)

for i in range(15):
    syn_label.narrow(0, i * num, num).fill_(label[i])

syn_feature = np.concatenate((syn_feature, real_feature_bg))
syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))

trainCls(syn_feature, syn_label, gan_epoch=1)

