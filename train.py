import os
import pathlib as p
import argparse
import tensorflow as tf
import process_data
import models

BUFFER_SIZE = 10000
CP_WEIGHTS_PATH = "/checkpoints/"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_train_ds(file_location):
    # Split datasets into parsed tweets and converted sentiment values
    tweets, sentiment = process_data.create_ds(file_location)
    t = tf.data.Dataset.from_tensor_slices(tweets)
    s = tf.data.Dataset.from_tensor_slices(sentiment)
    # Combine the tweets and Sentiment values into a single TensorFlow Dataset
    ts = tf.data.Dataset.zip((t, s))
    print("Created dataset of size: " + str(len(ts)))
    return ts


def create_test_train(train_location, test_location, buffer_size, batch_size):
    # Create train and test datasets from two file locations and creates single tensor batches of 10000 tensors
    # Shuffle helps preventing the network from getting stuck in a local minima
    # Prefetch improves speed of retrieval
    train_dataset = create_train_ds(train_location).shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = create_train_ds(test_location).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, test_dataset


def train_model(train_dataset, test_dataset, model, epochs, save_to):
    # Model checkpointing at maximum validation accuracy
    save_best_cp = tf.keras.callbacks.ModelCheckpoint(save_to + '/checkpoints/best_model',
                                                      monitor='val_accuracy',
                                                      mode='max',
                                                      verbose=1)

    # Callback to stop training if validation loss does no decrease after 3 itters.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Create Model using loss function of Binary Cross entropy and Adam Optimizer
    # Learning rate of 1e-4 and decay rate of 1e-6
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(1e-4, 1e-6),
                  metrics=['accuracy'])

    # Starts training
    train_hist = model.fit(train_dataset, epochs=epochs,
                           validation_data=test_dataset,
                           validation_steps=32,
                           callbacks=[early_stopping, save_best_cp])
    return model, train_hist


def test_model(test_dataset, save_to):
    # Load Model from checkpoints and evaluate
    m = tf.keras.models.load_model(save_to + '/checkpoints/best_model')
    loss, accuracy = m.evaluate(test_dataset)
    print('Test Loss:', loss)
    print('Test Accuracy:', accuracy)


def main(train_csv, test_csv, epochs, batch_sz, save_to):

    print("Starting training for " + train_csv + " and " + test_csv)
    print("Batch Size: " + str(batch_sz))
    # Create Datasets
    train_ds, test_ds = create_test_train(train_csv, test_csv, BUFFER_SIZE, batch_sz)
    print("Created Train of size: " + str(len(train_ds)) + " and test of size: " + str(len(test_ds)))
    # Create Model with encoded train dataset
    model = models.build_model_4(process_data.encode_text(train_ds))
    print("Model Built")
    # Start training, returns the model and training data
    model, hist = train_model(train_ds, test_ds, model, epochs, save_to)

    # Save Model at end of training
    model_structure = model.to_json()
    f = p.Path(str(p.Path(__file__).parent.resolve()) + "\\" + save_to + "/checkpoints/best_model/best_model.json")
    f.write_text(model_structure)
    print("Weights saved to: " + save_to + "/model")

    # Display and save training graph
    process_data.plot_training(hist, save_to)


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

i = 1
dir_to_save = "weights/" + str(i)
while os.path.exists(dir_to_save):
    i = i + 1
    dir_to_save = "weights/" + str(i)
os.mkdir(dir_to_save)
os.mkdir(dir_to_save + "/model")
main(args["train"], args["valid"], args["epochs"], args["batch"], dir_to_save)
