from helper import save_images

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import os

class InstantNGPEnv(gym.Env):
    def __init__(self, classifier_model, nerf_model, args, labels):
        super(InstantNGPEnv, self).__init__()

        self.nerf_model = nerf_model

        self.data_loaded, self.feature_grid = self.nerf_model.loadNeRFData()
        self.ground_truth_imgs = []

        self.classifier = classifier_model
        self.labels = labels
        self.negative_labels = args.negative_labels # Labels to avoid.
        self.target = args.target_class # Target class

        if self.target not in self.labels:
            self.labels.append(self.target)
        if args.true_class not in self.negative_labels:
            self.negative_labels.append(args.true_class)

        self.total_reward = 0
        self.epochs = 0
        
        self.Num_Imgs = args.num_imgs
        self.ImageWidth = args.image_width
        self.ImageHeight = args.image_height

        self.path_to_og_images = args.input_image_folder
        self.images_output_path = args.output_image_folder
        self.output_file = args.output_saved_nerf_file_path
        self.transforms_path = args.output_transforms_path

        self.theta_0 = args.theta_0
        self.theta_1 = args.theta_1
        self.theta_2 = args.theta_2

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(self.feature_grid.shape), dtype=np.float32)  # Define space of allowable feature grid modifications
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.Num_Imgs, self.ImageHeight, self.ImageWidth, 3), dtype=np.uint8) # Define state representation (e.g., observed images)


    def reset(self, seed=0):
        super().reset(seed=seed)
        
        list_dir = os.listdir(self.path_to_og_images)

        self.ground_truth_imgs = [Image.open(f"{self.path_to_og_images}{list_dir[i]}") for i in range(self.Num_Imgs)]
        self.ground_truth_imgs = np.array(self.ground_truth_imgs)

        self.data_loaded, self.feature_grid = self.nerf_model.loadNeRFData()

        return (self.ground_truth_imgs, {})
    
    def calc_mse(self, og_imgs, rendered_imgs):
        squared_diff = (og_imgs - rendered_imgs) ** 2
        mse = np.mean(squared_diff)

        # Returns the maximum MSE. Penalize the agent for high mse score.
        return mse.max()

    def get_reward(self, originalImage, renderedImage, predicted, all_classes):

        returned_reward = self.theta_2 * self.calc_mse(originalImage, renderedImage)

        top1_name = predicted[1] # Class for top 1.
        top1_conf = predicted[2] # Confidence in top 1.
        top1_nums = predicted[0] # Number of images classified as top 1.

        if top1_name not in self.negative_labels:

            if top1_name == self.target: # If target is predicted as top 1
                returned_reward += top1_conf * self.theta_0
            elif self.target in all_classes: # If target is predicted at all
                returned_reward += all_classes[self.target][0] # Rewards agent for confidence in true class.
            else: # Else, reward agent for predicting anything.
                returned_reward += top1_nums * .1
        else:
            returned_reward += self.theta_1 * top1_nums
        
        print("Step Reward: " + str(returned_reward) + " for " + predicted[1] + " with confidence: " + str(predicted[2]) + " and num: " + str((predicted[0])))
        return returned_reward

    def modify_imgs(self):
        noisy_imgs = []

        self.ground_truth_imgs = []
        generated = []
        for filename in os.listdir(self.images_output_path):
            generated.append(filename)
            self.ground_truth_imgs.append(Image.open(self.path_to_og_images + filename))
        self.ground_truth_imgs = np.array(self.ground_truth_imgs)

        for i in range(self.Num_Imgs):
            output_adv = Image.open(self.images_output_path + generated[i])

            noisy_imgs.append(output_adv)
        
        return noisy_imgs

    def pred_labels(self):
        pred_classes = {}

        noise_imgs = self.modify_imgs()
        preds_per_imgs, avg_target_conf = self.classifier.predict(noise_imgs, self.target, self.labels, top=1)

        for i in range(len(preds_per_imgs)): # The predictions produced by CLIP are in the same order as the noisey images we passed to it.
            pred = preds_per_imgs[i][0] # Grabs the tuple.

            # Add key-value pair to dictionary (key = label, value = [average confidence, number of images classified for this label]).
            if pred[1] not in pred_classes:
                pred_classes[pred[1]] = [pred[2], 1]
            else:
                pred_classes[pred[1]][0] *= pred_classes[pred[1]][1]
                pred_classes[pred[1]][0] += pred[2]
                pred_classes[pred[1]][1] += 1 # Number of images counter.
                pred_classes[pred[1]][0] /= pred_classes[pred[1]][1] # Average confidence

        maxKey = str(max(pred_classes, key=lambda x:pred_classes[x][1]))

        save_images(noise_imgs, self.images_output_path, maxKey, self.target, preds_per_imgs, pred_classes, self.negative_labels)
        reward = self.get_reward(self.ground_truth_imgs, np.array(noise_imgs), [pred_classes[maxKey][1], maxKey, pred_classes[maxKey][0]], pred_classes)  # Implement reward function

        print(f"Average confidence {avg_target_conf} in target class {self.target}")

        info = {
            "Target" : self.target,
            "Predicted" : maxKey,
            "Num_Imgs" : pred_classes[maxKey][1],
            "Confidence" : pred_classes[maxKey][0],
            "Epochs" : self.epochs
        }
        return noise_imgs, maxKey, pred_classes, reward, info

    def step(self, action):
        # Modify feature grid based on action
        self.feature_grid +=  action # Apply action to feature grid

        # Render image from modified feature grid
        self.nerf_model.SaveParameters(self.data_loaded, self.feature_grid)
        self.nerf_model.render_outputs(self.output_file, self.transforms_path, self.images_output_path, width=self.ImageHeight, height=self.ImageWidth)

        truncated, done, info = False, False, {}
        images, maxKey, pred_classes, reward, info = self.pred_labels()

        self.total_reward += reward

        # Setting the "done" flag is for the ".learn". Tells the program when it should stop.
        if (maxKey == self.target and pred_classes[maxKey][0] > .5) or (self.total_reward < -5):
            done = True
            self.total_reward = 0
            
            self.reset()

        self.epochs += 1

        print("info: " + str(info))
        print("Total Reward: " + str(self.total_reward))

        return images, reward, done, truncated, info