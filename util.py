import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    h, w = gray.shape[0], gray.shape[1]
    gray = np.reshape(gray, (h, w, 1))
    return gray

def rgb2gray_all(imgs):
	imgs_gray = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))

	for i in range(imgs.shape[0]):
		img = imgs[i,:,:,:]
		imgs_gray[i,:,:,:] = rgb2gray(img)
	return imgs_gray


def plots(imgs, figsize=(12,6), rows=1, title=None, titles=None):
    f = plt.figure(figsize=figsize)
    if title is not None: plt.title(title)
    for i in range(len(imgs)):
        sp = f.add_subplot(rows, len(imgs)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=8)
        plt.imshow(np.squeeze(imgs[i]), cmap='gray')