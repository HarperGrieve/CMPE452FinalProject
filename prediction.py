import numpy as np
import process_data
import argparse
from tensorflow.keras.models import load_model


def predict(string, model):
    return model.predict(np.array([process_data.pre_process(string)]))


def prediction(text, model):
    print(predict('covid is terrible i hate it', model))
    print(predict('I am not getting vaccinated its bad for you', model))
    print(predict('No more toilet paper. this is terrible', model))
    print(predict('The vaccine works i am happy', model))
    print(predict('Lockdown has ended. Im so exited for no masks', model))
    print(predict('Im so happy i can see my mom', model))


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="model location")
ap.add_argument("-t", "--tweet", type=str, required=True,
                help="tweet to predict")
args = vars(ap.parse_args())
model = load_model(args["model"])
prediction(args["tweet"], model)
