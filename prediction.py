import numpy as np
import process_data
import argparse
import tensorflow as tf
import sklearn.metrics as metrics


def predict(txt, m):
    # Predict from string
    pred = m.predict(np.array([process_data.pre_process(txt)]))
    if pred >= 0.5:
        return "Positive"
    else:
        return "Negative"


def get_metrics(y_t, y_p):
    print('Precision: ' + str(metrics.precision_score(y_t, y_p) * 100) + "%")
    print('Recall: ' + str(metrics.recall_score(y_t, y_p) * 100) + "%")

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="model location")
ap.add_argument("-t", "--tweet", type=str, required=True,
                help="tweet to predict")
args = vars(ap.parse_args())

# Create/Load model and datasets
model = tf.keras.models.load_model(args["model"])
x_test, y_true = process_data.create_ds("Corona_NLP_test.csv")
# Evaluate model
loss, accuracy = model.evaluate(x_test, y_true)
y_test = y_true.numpy()
y_pred = tf.cast(tf.greater(model.predict(x_test), 0.5), tf.int32).numpy()
print('Test Loss:', loss)
print('Test Accuracy: ' + str(accuracy*100) + "%")
get_metrics(y_test, y_pred)

t1 = "Covid made me loose my job, now i cannot eat"
print(t1)
print("Sentiment Predition: " + predict(t1, model))
t2 = "Im so happy covid is ending, I cant wait to see my friends"
print(t2)
print("Sentiment Predition: " + predict(t2, model))


# prediction(args["tweet"], model)
