from tensorflow.keras.models import model_from_json
import codecs
import itertools
import numpy as np
import cv2 as cv
import sys
import os


DEFAULT_PATH = 'division_directory'
""" Input file directory for the license plate"""
DEFAULT_VOCABULARY_FILE = 'pretrained/vocabulary.txt'
"""A list of characters that can be recognised"""

DEFAULT_MODEL_FILE = "pretrained/model.predict.json"
"""A JSON file containing the model definition."""

DEFAULT_WEIGHTS = "pretrained/weights.best.h5"
"""The path to the exported weights for the model."""

DEFAULT_DETECTION_THRESHOLD = 0.75
"""The accuracy of the result that must be passed before reporting the result."""

VISUALISATION_DIR = "visualisations"
"""Directory used to store visualization of images."""

#! the txt file where we'll save the plates text to in order.
OUTPUT_TXT_FILE = "inference/output/plates.txt"


def init_model_and_vocab(
    model_path=DEFAULT_MODEL_FILE,
    weights_path=DEFAULT_WEIGHTS,
    vocabulary_file_path=DEFAULT_VOCABULARY_FILE,
):
    #! Intializing The Model
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    #! Intializing The Vocabulary
    vocab = load_vocabulary_file(vocabulary_file_path)

    return model, vocab


def load_vocabulary_file(file_path):

    with codecs.open(file_path, encoding="utf-8") as f:
        content = [line.strip() for line in f]
        return content


def indexes_to_string(label_indexes, classes):
    ret = []
    for index in label_indexes:
        if index == len(classes):  # CTC Blank
            ret.append("")
        else:
            ret.append(classes[index])
    return "".join(ret)


def save_results_to_txt_file(plate_text):
    global OUTPUT_TXT_FILE
    num_of_lines = 0

    #! Count how many line in the file
    with open(OUTPUT_TXT_FILE, "r", encoding="UTF-8") as txt:
        for line in txt.readlines():
            num_of_lines += 1

    #! append new plates texts
    with open(OUTPUT_TXT_FILE, "a", encoding="UTF-8") as txt:
        upper, lower = plate_text.split(" <==> ")
        txt.write(
            f"#{num_of_lines+1}\t\tUpper: {upper}\t\tLower: {lower}\n")
        num_of_lines += 1


def best_path_prediction(prediction, classes):
    ret = []

    for j in range(prediction.shape[0]):
        out_best = list(np.argmax(prediction[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        pred = indexes_to_string(out_best, classes)
        ret.append(pred)
    return "".join(ret)


def preprocess_model_input(bgr_images, model):
    #! model input shape is  (None, 32, None, 3)
    _, height, width, channels = model.input_shape

    model_input = []
    for image in bgr_images:
        try:
            if channels == 1:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            else:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        except Exception as e:
            print(f"\n\n[THREADING ERROR]: {e}")
            print(f"\nIMAGE SHAPE: {image.shape}\n\n")
            continue

        #! resize the image to model expected dimensions
        if width is None:
            img_height, img_width = image.shape[:2]
            ratio = float(height) / img_height
            width = int(ratio * img_width)

        image = cv.resize(image, (width, height))

        #! add the image to the list
        model_input.append(image)

    #! 1) put a new axis for the image
    #! 2) Normalize the image to give better results
    return np.array(model_input, dtype='float32').reshape((-1, height, width, channels)) / 255.0


def predict_characters(
    model,  # ! object
    vocabulary,  # ! list
    #! each one of these images is a numpy array. every 2 consecutive elements
    #! forms a single plate where the first part is the upper-half and other is the lowerr-half
    bgr_images,
    output_array
):

    number_of_inputs = len(bgr_images)
    if number_of_inputs < 2:
        print("\n\n[THREADING]: NOT ENOUGH INPUT IMAGE\n\n")
        return

    model_input = preprocess_model_input(bgr_images, model)
    if len(model_input) == 0:  # images are corrupted
        print("\n\n[THREADING]: ERROR IN MODEL INPUT\n\n")
        sys.exit()

    predictions = model.predict(model_input)

    #! will have the same number of items as the input images
    plate_text = None
    sep = " <==> "
    #!"predictions" shape is (number_of_input_images, number_of_boxes_found_in_image, number_of_classes=245)
    for i, pred in enumerate(predictions):
        bgr_images.pop(0)  # ! remove the recognized image from history

        #! we must add a new dimension because the "best_path_prediction" method
        #! expects the predictions to be 3 dimensions
        text = best_path_prediction(pred[None, ...], vocabulary)

        if (i % 2 == 0):  # ! plate upper-half text
            plate_text = text
        else:  # ! plate lower-half text
            plate_text += (sep + text)
            save_results_to_txt_file(plate_text)
            output_array.append(plate_text.split(sep))

    sys.exit()  # ! to kill the thread when finished
