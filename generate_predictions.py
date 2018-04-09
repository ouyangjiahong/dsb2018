import os
import sys

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

if __name__=="__main__":
    IMG_HEIGHT = IMG_WIDTH= 256
    IMG_CHANNELS = 1

    csv_save_name = 'data/unet_submission.csv'
    pred_path = 'data/test_prediction.npz'
    test_path = 'data/stage1_test/'

    # load prediction
    print('Loading prediction')
    file = np.load(pred_path)
    prediction = file['result']

    test_ids = next(os.walk(test_path))[1]
    # Create list of upsampled test masks
    sizes_test = []
    print('Getting test image sizes ...')
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = test_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])

    preds_test_upsampled = []
    for i in range(len(prediction)):
        pred = prediction[i]
        preds_test_upsampled.append(resize(np.squeeze(pred),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    # Create submission file
    new_test_ids = []
    rles = []
    for n, id_ in tqdm(enumerate(test_ids)):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    sub.to_csv(csv_save_name, index=False)
    print('Saved submission file to ', csv_save_name)
