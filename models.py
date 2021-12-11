import tensorflow as tf


def build_model_1(encoder):
    return tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 32, mask_zero=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, bias_initializer='zeros')
    ])


def build_model_2(encoder):
    return tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 32, mask_zero=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, bias_initializer='zeros')
    ])


def build_model_3(encoder):
    return tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, bias_initializer='zeros')
    ])


def build_model_4(encoder):
    return tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, bias_initializer='zeros')
    ])
