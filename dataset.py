import numpy as np
import torch
# import pandas as pd 
from torch.utils.data import Dataset
from util import *

import os.path
from os import path


class FeaturesCls(Dataset):

    def __init__(self, opt, features=None, labels=None, val=False, split='seen', classes_to_train=None):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        self.classes_to_train = classes_to_train
        self.classid_tolabels = None
        self.features = features
        self.labels = labels
        if self.classes_to_train is not None:
            self.classid_tolabels = {label: i for i, label in enumerate(self.classes_to_train)}

        print(f"class ids for unseen classifier {self.classes_to_train}")
        if 'test' in split:
            self.loadRealFeats(syn_feature=features, syn_label=labels, split=split)

    def loadRealFeats(self, syn_feature=None, syn_label=None, split='train'):
        if 'test' in split:
            # voc
            # self.features = np.load(f"/media/dxm/D/DXM/Feature/VOC/test_0.7_0.3_feats.npy")
            # self.labels = np.load(f"/media/dxm/D/DXM/Feature/VOC/test_0.7_0.3_labels.npy")

            # coco
            self.features = np.load(f"/media/dxm/D/DXM/Feature/E12/test_0.6_0.3_feats.npy")
            self.labels = np.load(f"/media/dxm/D/DXM/Feature/E12/test_0.6_0.3_labels.npy")

            print(f"{len(self.labels)} testsubset {self.opt.testsplit} features loaded")
            # import pdb; pdb.set_trace()

    def replace(self, features=None, labels=None):
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        print(f"\n=== Replaced new batch of Syn Feats === \n")

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
        if self.classid_tolabels is not None:
            batch_label = self.classid_tolabels[batch_label]
        return batch_feature, batch_label

    def __len__(self):
        return len(self.labels)


class FeaturesGAN():
    def __init__(self, opt):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        # self.attribute = np.load(opt.class_embedding)

        print("loading numpy arrays")
        self.all_features = np.load(self.opt.datafeature_root)
        self.all_labels = np.load(self.opt.datalabel_root)
        if self.all_features.shape[0] != len(self.all_labels):
            self.all_features = self.all_features.reshape(len(self.all_labels),
                                                          len(self.all_features) // len(self.all_labels))
        # part_feature = np.load(f'/home/disk/DXM/Feature/E12/train_0.7_0.3_23_feats.npy')
        # part_label = np.load(f'/home/disk/DXM/Feature/E12/train_0.7_0.3_23_labels.npy')
        # self.all_features = np.append(self.all_features, part_feature)
        # self.all_labels = np.append(self.all_labels, part_label)
        print(f'loaded data from {self.opt.trainsplit}')
        self.pos_inds = np.where(self.all_labels > 0)[0]
        self.neg_inds = np.where(self.all_labels == 0)[0]

        self.unique_labels = np.unique(self.all_labels)
        self.num_bg_to_take = len(self.pos_inds) // len(self.unique_labels)

        print(f"loaded {len(self.pos_inds)} fg labels")
        print(f"loaded {len(self.neg_inds)} bg labels ")
        print(f"bg indexes for each epoch {self.num_bg_to_take}")

    def epochData(self, include_bg=False):
        fg_inds = np.random.permutation(self.pos_inds)
        inds = np.random.permutation(fg_inds)[:int(self.opt.gan_epoch_budget)]
        if include_bg:
            bg_inds = np.random.permutation(self.neg_inds)[:self.num_bg_to_take]
            inds = np.random.permutation(np.concatenate((fg_inds, bg_inds)))[:int(self.opt.gan_epoch_budget)]
        features = self.all_features[inds]
        labels = self.all_labels[inds]
        return features, labels

    def getBGfeats(self, num=1000):
        bg_inds = np.random.permutation(self.neg_inds)[:num]
        print(f"{len(bg_inds)} ")
        return self.all_features[bg_inds], self.all_labels[bg_inds]

    # def getBG_and_seen_feat(self, bg=300, seen=5):
    #
    #     all_inds = np.random.permutation(self.neg_inds)[:bg]
    #     for i in self.unique_labels[1:]:
    #         num = np.where(self.all_labels == i)[0]
    #         inds = np.random.permutation(num)[:seen]
    #         all_inds = np.concatenate((all_inds, inds))
    #     labels = np.zeros(bg+seen*65)
    #     print(f"background: {bg}, seen: {seen*65}")
    #     return self.all_features[all_inds], labels

    def getBG_and_seen_feat(self, bg=300, seen=300):
        bg_inds = np.random.permutation(self.neg_inds)[:bg]
        fg_inds = np.random.permutation(self.pos_inds)[:seen]
        all_inds = np.concatenate((fg_inds, bg_inds))
        labels = np.zeros(bg+seen)
        print(f"background: {bg}, seen: {seen}")
        return self.all_features[all_inds], labels

    def __len__(self):
        return len(self.all_labels)


