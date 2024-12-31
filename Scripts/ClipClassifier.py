import clip
from torch import no_grad
from torch import stack
from torch import cat
from PIL import Image
import numpy as np

class CLIP_Classifier:
    def __init__(self, device):

        # Load the model
        self.device = device
        self.model, self.preprocess = clip.load('RN50', self.device)

    def predict(self, noise_imgs, target_class, classes, top):
        image_input = []
        for image in noise_imgs:
           image_input.append(self.preprocess(image))
        # Prepare the inputs
        image_input = stack(image_input).to(self.device)
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
        for similarity in similarities:
          values, indices = similarity.softmax(dim=-1).topk(top)
          predictions = []
          for value, index in zip(values, indices):
            predictions.append(("", classes[index], value.item()))
          per_image_predictions.append(predictions)

        target_confs = []
        for similarity in similarities:
           values, indices = similarity.softmax(dim=-1).topk(len(classes))
           for value, index in zip(values, indices):
              if classes[index] == target_class:
                 target_confs.append(value.item())
        return per_image_predictions, np.array(target_confs).mean()

    def predict_single(self, noise_img, target_class, classes, top):
        # Prepare the inputs
        image = noise_img
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
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
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(top)

        # Print the result
        # print("\nTop predictions:\n")
        target_pred = 0
        predictions = []
        for value, index in zip(values, indices):
            predictions.append(("", classes[index], value.item()))

            if classes[index] == target_class:
               target_pred = value.item()
            # print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
            # print("value: " + str(value.item()))
        return predictions, target_pred