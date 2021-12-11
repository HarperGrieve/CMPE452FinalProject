import nltk
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

VOCAB_SZ = 3000


def create_ds(file_location):
    # Load CSV file and drop unneeded columns
    csv = pd.read_csv(file_location)
    csv = csv.drop(columns=['UserName', 'ScreenName', 'Location', 'TweetAt'])
    # Convert Sentiment from strings to binary, remove any tweets with neutral sentiment
    csv['Sentiment'] = convert_sentiment(csv['Sentiment'])
    csv = csv[csv['Sentiment'] != -1]
    # Process the tweets
    for tweet in range(len(csv['OriginalTweet'])):
        csv['OriginalTweet'].iloc[tweet] = pre_process(csv['OriginalTweet'].iloc[tweet])
    # Return separated tweets and sentiment values
    tweets = tf.constant(csv['OriginalTweet'].to_numpy(), dtype=tf.string)
    sentiment = tf.constant(csv['Sentiment'].to_numpy(), dtype=tf.int32)
    return tweets, sentiment


def convert_sentiment(sentiment):
    # Convert string to binary equivalent
    for x in range(len(sentiment)):
        if sentiment[x] == 'Extremely Positive' or sentiment[x] == 'Positive':
            sentiment[x] = 1
        elif sentiment[x] == 'Negative' or sentiment[x] == 'Extremely Negative':
            sentiment[x] = 0
        elif sentiment[x] == 'Neutral':
            sentiment[x] = -1
    return sentiment


def pre_process(line):
    new_words = []
    # Initialize the lemmatizer
    lem = nltk.WordNetLemmatizer()
    # Iterate through tweet word by word
    for word in line.split():
        # Remove any words containing http, www, or @ symbol
        if not (word == '' or word.startswith('http') or word.startswith('www') or word.startswith('@')):
            # Remove hashtag symbol to leave word
            if word.startswith('#'):
                word = word[1:]
            # Remove all non alpha-numeric characters, make all letters lowercase
            word = ''.join(e for e in word if e.isalnum())
            word = word.lower()
            # Apply lemmatization to words
            if word != '':
                word = lem.lemmatize(word)
                new_words.append(word)
    # join and return tweet
    return bytes(' '.join(new_words), 'utf-8')


def encode_text(train):
    # Use text vectorization to create dictionary of train set
    e = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SZ)
    e.adapt(train.map(lambda text, label: text))
    return e


def plot_graphs(history, metric):
    # Create training graphs
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def plot_training(history, save_to):
    # Create training graphs
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.savefig(save_to + "/training.png")
    plt.show()
