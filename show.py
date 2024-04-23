import numpy as np
from PIL import Image
arr_0 = np.load('stable-diff/all_300*15_512.npy')

# arr_1 = np.load('Guided-Result/img/label.npy')

for i in range(arr_0.shape[0]):
    img = Image.fromarray(arr_0[i])
    img.save(f"../stable_img/{i}.png")