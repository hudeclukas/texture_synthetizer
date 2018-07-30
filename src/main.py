import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from data_streamer import data_streamer
from network import texture_generator_network as tng

PATH_2_TEXDAT = "D:/Vision_Images/Pexels_textures/TexDat/official"
MODEL_NAME = "texture_synthesis_180728"

MAX_ITERS = 25001
BATCH_SIZE = 20
IMAGE_SIZE = (150,150,1)

TRAIN = True

def main():
    ds = data_streamer(PATH_2_TEXDAT)
    print("Starting loading data...")
    start = time.time() * 1000
    ds.read_train_and_test()
    print((time.time() * 1000) - start, "ms")


    generator = tng(IMAGE_SIZE)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(
        loss=generator.loss,
        global_step=global_step
    )

    save_dir = 'model/' + MODEL_NAME + '/'
    saver = tf.train.Saver(max_to_keep=10)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_ckpt = os.path.join(save_dir, 'checkpoint')

    with tf.Session() as sess:
        if os.path.exists(model_ckpt):
            # restore checkpoint if it exists
            try:
                print("Trying to restore last checkpoint ...")
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
                saver.restore(sess, save_path=last_chk_path)
                print("Restored checkpoint from:", last_chk_path)
            except:
                print("Failed to restore checkpoint. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        if TRAIN:
            writer = tf.summary.FileWriter('logs/' + MODEL_NAME + '/', graph=sess.graph)
            merge_summary = tf.summary.merge_all()
            for epoch in range(0, 6):
                print("Epoch {:01d}".format(epoch))

                for step in range(0, MAX_ITERS):
                    step = MAX_ITERS * epoch + step
                    masked, originals = ds.random_masked_textures(BATCH_SIZE, 20, 70, True)
                    l_rate = 0.002 / (1.75 * float(epoch + 1))

                    board_summary, _, loss_v = sess.run(
                        [merge_summary, train_step, generator.loss], feed_dict={
                            generator.input_batch: masked,
                            generator.originals: originals,
                            generator.dropout_keep_prob: 0.6,
                            learning_rate: l_rate
                        })

                    if step < 50 or step % 10 == 0:
                        print("Step: [{:04d}.] --> loss: |{:3.8f}|...".format(step, loss_v))

                    if step < 50 or step % 100 == 0:
                        writer.add_summary(board_summary, step)



if __name__ == "__main__":
    main()