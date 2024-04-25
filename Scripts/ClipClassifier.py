import clip
from torch import no_grad
from torch import stack
from torch import cat
from PIL import Image
import os
import numpy as np

class CLIP_Classifier:
    def __init__(self, device):

        # Load the model
        self.device = device
        self.model, self.preprocess = clip.load('RN50', self.device)

    def predict(self, directory, num_imgs, classes, top):
        images = []
        image_input = []
        list_dir = os.listdir(directory)
        for i in range(num_imgs):
           image = Image.open(f"{directory}{list_dir[i]}")
           images.append(np.asarray(image))
           image_input.append(self.preprocess(image))
        # Prepare the inputs
        image_input = stack(image_input).to(self.device)
        # image_input = self.preprocess(images).unsqueeze(0).to(self.device)
        text_inputs = cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(self.device)

        if top > len(classes):
            print("The number of top predictions you want is higher than the number of classes provided.")
            return []

        # Calculate features
        with no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).squeeze(0)

        per_image_predictions = []
        count = 0
        for similarity in similarities:
          values, indices = similarity.softmax(dim=-1).topk(top)
          predictions = []
          for value, index in zip(values, indices):
            predictions.append(("", classes[index], value.item()))
          per_image_predictions.append(predictions)
          count += 1
        return per_image_predictions, images