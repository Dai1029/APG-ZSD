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

start_epoch = 0

seenDataset = FeaturesGAN(opt)

trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, gen_type='FG')

if opt.netD and opt.netG:
    start_epoch = trainFGGAN.load_checkpoint()

a = opt.bl
background = int(opt.syn_num*2*a)
seen_num = opt.syn_num*2 - background
print(f"background{background}:seen{seen_num}")

for epoch in range(start_epoch, opt.nepoch):
    # features, labels = seenDataset.epochData(include_bg=False)
    features, labels = seenDataset.epochData(include_bg=True)
    # train GAN
    trainFGGAN(epoch, features, labels, type=0)
    if epoch >= 100:
        # synthesize features
        syn_feature, syn_label = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, num=opt.syn_num)
        # dxm
        # add seen class as background. background:seen
        # real_feature_bg, real_label_bg = seenDataset.getBG_and_seen_feat(600, 0)

        real_feature_bg, real_label_bg = seenDataset.getBG_and_seen_feat(background, seen_num)

        # concatenate synthesized + real bg features85
        syn_feature = np.concatenate((syn_feature.data.numpy(), real_feature_bg))
        syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))

        trainCls(syn_feature, syn_label, gan_epoch=epoch)

