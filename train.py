import os
import time

import pandas as pd
import argparse
import tensorflow as tf
import process_data
import matplotlib.pyplot as plt
from datetime import datetime

BUFFER_SIZE = 10000
CP_WEIGHTS_PATH = "/checkpoints/CMPE452_FP_{epoch:04d}.ckpt"


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


def create_ds(file_location):
    csv = pd.read_csv(file_location)
    csv = csv.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])
    csv = process_data.convert_sentiment(csv)
    for tweet in range(len(csv['OriginalTweet'])):
        csv['OriginalTweet'][tweet] = process_data.pre_process(csv['OriginalTweet'][tweet])
    tweets = tf.constant(csv['OriginalTweet'].to_numpy(), dtype=tf.string)
    sentiment = tf.constant(csv['Sentiment'].to_numpy(), dtype=tf.int32)
    t = tf.data.Dataset.from_tensor_slices(tweets)
    s = tf.data.Dataset.from_tensor_slices(sentiment)
    ts = tf.data.Dataset.zip((t, s))
    print("Created dataset of size: " + str(len(ts)))
    return ts


def encode_text(train):
    e = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SZ)
    e.adapt(train.map(lambda text, label: text))
    return e


def build_model(encoder):
    return tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])


def create_test_train(train_location, test_location, buffer_size, batch_size):
    train_dataset = create_ds(train_location).shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = create_ds(test_location).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("Created " + str(len(train_dataset)) + " training batches and " + str(len(test_dataset)) + "test batches")
    return train_dataset, test_dataset


def plot_training(history):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.show()


def train_model(train_dataset, test_dataset, model, epochs, batch_sz, save_to):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_to + CP_WEIGHTS_PATH,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * batch_sz)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    train_hist = model.fit(train_dataset, epochs=epochs,
                           validation_data=test_dataset,
                           batch_sz=batch_sz,
                           validation_steps=30,
                           callbacks=[early_stopping, checkpoint])

    return model, train_hist


def train(train_csv, test_csv, epochs, batch_sz, save_to):
    print("Starting training for " + train_csv + " and " + test_csv)
    print("Batch Size: " + str(batch_sz))
    train_ds, test_ds = create_test_train(train_csv, test_csv, BUFFER_SIZE, batch_sz)
    print("Created Train of size: " + str(len(train_ds)) + " and test of size: " + str(len(test_ds)))
    model = build_model(encode_text(train_ds))
    print("Build Model")
    model.save_weights(CP_WEIGHTS_PATH.format(epoch=0))
    model, hist = train_model(train_ds, test_ds, model, epochs, batch_sz, save_to)
    model.save(save_to + "/CMPE452_FP")
    print("Weights saved to: " + save_to + "/CMPE452_FP")
    plot_training(hist)

    loss, accuracy = model.evaluate(test_ds)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", type=str, required=True,
                help="path to train.csv")
ap.add_argument("-v", "--valid", type=str, required=True,
                help="path to test.csv")
ap.add_argument("-e", "--epochs", type=int, required=True,
                help="how many epochs")
ap.add_argument("-w", "--vocab", type=int, required=True,
                help="vocabulary size")
ap.add_argument("-b", "--batch", type=int, required=True,
                help="batch size")
args = vars(ap.parse_args())
VOCAB_SZ = args["vocab"]
i = 1
dir_to_save = "weights/" + str(i)
while os.path.exists(dir_to_save):
    i = i + 1
    dir_to_save = "weights/" + str(i)
os.mkdir(dir_to_save)
train(args["train"], args["valid"], args["epochs"], 64, dir_to_save)
