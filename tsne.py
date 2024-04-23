import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from splits import get_unseen_class_ids

unseen = get_unseen_class_ids('coco', '65_15')

real_f = np.load("/media/dxm/D/DXM/Feature/E12/test_0.6_0.3_feats.npy")
real_l = np.load("/media/dxm/D/DXM/Feature/E12/test_0.6_0.3_labels.npy")

fg_inds = np.random.permutation(range(300))[:100]

# fake_f = np.load('Diff_Feature/Unseen_1024_RN50/hot.npy')
fake_f = np.load('Diff_Feature/Extract_mm/all.npy')

X = fake_f[fg_inds]
y = np.repeat([10], 100)
for i in range(4):
    X = np.concatenate((X, fake_f[fg_inds+300*(i+1)]), 0)
    y = np.concatenate((y, np.repeat([i+11], 100)), 0)
#
# fake_f = np.load('Diff_Feature/Unseen_1024_RN50/mouse.npy')
# X = np.concatenate((X, fake_f[fg_inds]), 0)
# y_i = np.repeat([12], 30)
# y = np.concatenate((y, y_i), 0)
#
# fake_f = np.load('Diff_Feature/Unseen_1024_RN50/toaster.npy')
# X = np.concatenate((X, fake_f[fg_inds]), 0)
# y_i = np.repeat([13], 30)
# y = np.concatenate((y, y_i), 0)
#
# fake_f = np.load('Diff_Feature/Unseen_1024_RN50/hair.npy')
# X = np.concatenate((X, fake_f[fg_inds]), 0)
# y_i = np.repeat([14], 30)
# y = np.concatenate((y, y_i), 0)

for i in range(5):
    pos_inds = np.where(real_l == unseen[i+10])[0]
    if len(pos_inds) < 100:
        fg_inds = np.random.permutation(pos_inds)
        X = np.concatenate((X, real_f[fg_inds]), 0)
        y_i = np.repeat([i+10], len(pos_inds))
        y = np.concatenate((y, y_i), 0)
    else:
        fg_inds = np.random.permutation(pos_inds)[:100]
        X = np.concatenate((X, real_f[fg_inds]), 0)
        y_i = np.repeat([i+10], 100)
        y = np.concatenate((y, y_i), 0)


'''X是特征，不包含target;X_tsne是已经降维之后的特征'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(20, 20))
size = 70
size_real = 100
for i in range(X_norm.shape[0]):
    # if i < 30*5:
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color="green", fontdict={'weight': 'bold', 'size': 9})
    # else:
    #     if y[i] >= 10:
    #         plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color="blue", fontdict={'weight': 'bold', 'size': 9})
    #     else:
    #         plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color="gray", fontdict={'weight': 'bold', 'size': 9})
    if i < 100*5:
        if y[i] == 10:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size, marker='*', color="red", alpha=0.65)
        elif y[i] == 11:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size, marker='*', color="blue", alpha=0.65)
        elif y[i] == 12:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size, marker='*', color="yellow", alpha=0.65)
        elif y[i] == 13:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size, marker='*', color="magenta", alpha=0.65)
        else:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size, marker='*', color="green", alpha=0.65)
    else:
        if y[i] == 10:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size_real, color="red", alpha=0.65)
        elif y[i] == 11:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size_real, color="blue", alpha=0.65)
        elif y[i] == 12:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size_real, color="yellow", alpha=0.65)
        elif y[i] == 13:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size_real, color="magenta", alpha=0.65)
        elif y[i] == 14:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size_real, color="green", alpha=0.65)
        else:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=size_real, color="gray", alpha=0.65)
plt.xticks([])
plt.yticks([])
plt.show()