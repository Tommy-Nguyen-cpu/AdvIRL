import numpy as np
import pandas as pd

from PIL import Image

import os
import json
import copy

def baseline_classification(classifier_model, input_image_folder, labels, target_class, true_class):
    """
    Classify images of the object prior to applying adversarial noise.

    Args:
        classifier_model: classifier to predict baseline classification.
        input_image_folder: Path to the folder containing original images of the object.
        labels: list of strings containing all labels to predict.
        target_class: Target class for our adversarial attack.
        true_class: The correct classification for the object.
    """

    df = pd.DataFrame(columns=["Image Name", "Class", "Confidence"])
    for name in os.listdir(input_image_folder):
        image = Image.open(input_image_folder + name)
        pred = classifier_model.predict_single(image, target_class, labels, top = 1)[0][0]
        df.loc[-1] = [name, pred[1], pred[2]]
        df.index = df.index + 1
        df = df.sort_index()
    df.to_csv(input_image_folder + f"../original_predictions_{true_class}.csv")

def generate_transforms(og_transforms_path, output_transforms_path, number_of_transforms = 10):
    """
    Modify NeRF msgpack so we only generate a select few transforms of the scene.

    Args:
        og_transforms_path: Path to the original transforms.json file.
        output_transforms_path: Path to the output transforms.json file that contains selected transforms.
        number_of_transforms: Determines how many transforms are included in the output transforms.json file.
    """
    file = open(og_transforms_path)
    transforms = json.load(file)
    copyJson = copy.deepcopy(transforms)
    with open(output_transforms_path, "w") as short_output:
        copyJson['frames'] = list(np.array(copyJson['frames'])[:number_of_transforms])
        json.dump(copyJson, short_output)