import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from data_streamer import data_streamer

PATH_2_TEXDAT = "D:/Vision_Images/Pexels_textures/TexDat/exp"
MODEL_NAME = "texture_synthesis_180728"

MAX_ITERS = 25001
BATCH_SIZE = 24

TRAIN = True

def main():
    ds = data_streamer(PATH_2_TEXDAT)
    print("Starting loading data...")
    start = time.time() * 1000
    ds.read_train_and_test()
    print((time.time() * 1000) - start, "ms")

    masked, original = ds.random_masked_textures(10,20,70,True)

    for i in range(len(masked)):
        plt.imshow(masked[i].reshape((150,150)),cmap="gray")
        plt.show()

if __name__ == "__main__":
    main()