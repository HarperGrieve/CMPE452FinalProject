import pandas as pd
import nltk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SZ = 1000

TRAIN_LOCATION = "C:\\Users\\harpe\\school\\CMPE452\\FinalProjectDataset\\archive\\Corona_NLP_test.csv"
TEST_LOCATION = "C:\\Users\\harpe\\school\\CMPE452\\FinalProjectDataset\\archive\\Corona_NLP_test.csv"


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


def convert_sentiment(sentiment):
    for x in range(len(sentiment)):
        if sentiment[x] == 'Extremely Positive' or sentiment[x] == 'Positive' or sentiment[x] == 'Neutral':
            sentiment[x] = 1
        elif sentiment[x] == 'Negative' or sentiment[x] == 'Extremely Negative':
            sentiment[x] = 0
    return sentiment


def pre_process(line):
    new_words = []
    lem = nltk.WordNetLemmatizer()
    for word in line.split():
        if not (word == '' or word.startswith('http') or word.startswith('www') or word.startswith('@')):
            if word.startswith('#'):
                word = word[1:]
            word = ''.join(e for e in word if e.isalnum())
            word = word.lower()
            if word != '':
                word = lem.lemmatize(word)
                new_words.append(word)
    return bytes(' '.join(new_words), 'utf-8')


def create_ds(file_location):
    csv = pd.read_csv(file_location)
    csv = csv.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])
    csv['Sentiment'] = convert_sentiment(csv['Sentiment'])
    for tweet in range(len(csv['OriginalTweet'])):
        csv['OriginalTweet'][tweet] = pre_process(csv['OriginalTweet'][tweet])
    tweets = tf.constant(csv['OriginalTweet'].to_numpy(), dtype=tf.string)
    sentiment = tf.constant(csv['Sentiment'].to_numpy(), dtype=tf.int32)
    t = tf.data.Dataset.from_tensor_slices(tweets)
    s = tf.data.Dataset.from_tensor_slices(sentiment)
    return tf.data.Dataset.zip((t, s))


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
    return train_dataset, test_dataset


def plot_training(history):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.show()


def train_model(train_dataset, test_dataset, model):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    hist = model.fit(train_dataset, epochs=20,
                     validation_data=test_dataset,
                     validation_steps=30,
                     callbacks=[callback])

    plot_training(hist)

    test_loss, test_acc = model.evaluate(test_dataset)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    return model


def predict(string, model):
    return model.predict(np.array([pre_process(string)]))


def main():
    train, test = create_test_train(TRAIN_LOCATION, TEST_LOCATION, BUFFER_SIZE, BATCH_SIZE)
    model = build_model(encode_text(train))
    model = train_model(train, test, model)
    print(predict("covid is bad", model))
    print(predict("covid is good", model))


if __name__ == '__main__':
    main()
