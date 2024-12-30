import numpy as np
import pandas as pd

from PIL import Image

import os
import json
import copy
import datetime

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

def save_images(adv_images : list[Image.Image], images_output_path : str, top_label : str, target_label : str, prediction_per_images : list[tuple[str, str, float]], predicted_classes, negative_labels : list[str]):
    """
    Saves adversarial images rendered by AdvIRL. Only saves images if the top label predicted is not found in our list of negative labels to avoid, our target label is in the predicted class, and if at least 7 images are predicted as the target class.

    Args:
        adv_images: List of rendered adversarial images.
        images_output_path: Path that will hold our output images (subfolder will contain all successfully misclassified images).
        top_label: The label most consistently misclassified across all of our adversarial images.
        target_label: The label we want our adversarial images to be misclassified as.
        prediction_per_images: A list containing the classification of each image in our "adv_images" list.
        predicted_classes: Dictionary containing the average labels and confidences predicted across all images. Key = label, value = list containing average confidence and number of images predicted as label.
        negative_labels: List of labels we want to avoid.
    """
    if top_label not in negative_labels and target_label in predicted_classes and predicted_classes[target_label][1] > 6:
                saved_folder = datetime.datetime.now().strftime("%I%M%p%S on %B %d %Y")
                os.makedirs(images_output_path + "../" + saved_folder)
                for i in range(len(prediction_per_images)):
                    pred_tuple = prediction_per_images[i][0]
                    pred_label = pred_tuple[1]
                    confidence = pred_tuple[2]
                    if pred_label not in negative_labels:
                        adv_images[i].save(images_output_path +"../"+ saved_folder + f"/{pred_label}_{confidence}_{i}_{adv_images[i].filename[adv_images[i].filename.rindex('/')+1:]}")
