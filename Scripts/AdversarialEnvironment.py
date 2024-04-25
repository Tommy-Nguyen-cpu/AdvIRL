import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import os
import random

class InstantNGPEnv(gym.Env):
    def __init__(self, classifier_model, nerf_model, num_of_imgs, image_shape, true_class, target_class, path_to_og_images, image_output_path, 
                 output_saved_file, transforms_path, labels, logPath = '../AdvOutput/rewards.log'):
        super(InstantNGPEnv, self).__init__()

        # Load Instant-NGP and MobileNetV2 models
        self.nerf_model = nerf_model

        self.data_loaded, self.feature_grid = self.nerf_model.loadNeRFData()
        self.ground_truth_img = None
        self.log = open(logPath, 'a')


        self.mobilenetv2 = classifier_model  # Load MobileNetV2 model
        self.labels = labels
        self.true_class = true_class # True class
        self.target = target_class # Target class
        self.total_reward = 0
        self.epochs = 0
        
        self.Num_Imgs = num_of_imgs
        self.ImageWidth = image_shape[0]
        self.ImageHeight = image_shape[1]

        self.path_to_og_images = path_to_og_images
        self.images_output_path = image_output_path
        self.output_file = output_saved_file
        self.transforms_path = transforms_path

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(self.feature_grid.shape), dtype=np.float32)  # Define space of allowable feature grid modifications
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.Num_Imgs, self.ImageHeight, self.ImageWidth, 3), dtype=np.uint8) # Define state representation (e.g., feature grid, image)


    def reset(self, seed=0):
        super().reset(seed=seed)
        # Reset feature grid or load new one
        self.ground_truth_img = np.empty((0, self.ImageHeight, self.ImageWidth, 3), dtype=np.uint8)
        
        list_dir = os.listdir(self.path_to_og_images)
        for i in range(0, self.Num_Imgs):
            image = Image.open(f"{self.path_to_og_images}{list_dir[i]}")

            self.ground_truth_img = np.concatenate((self.ground_truth_img, np.asarray(image)[None, ...]), axis=0)


        self.data_loaded, self.feature_grid = self.nerf_model.loadNeRFData()

        return (self.ground_truth_img, {})
    
    def calc_mse(self, og_imgs, rendered_imgs):
        squared_diff = (og_imgs - rendered_imgs) ** 2
        mse = np.mean(squared_diff)

        # Returns the maximum MSE. Penalize the agent for high mse score.
        return mse.max()

    def get_reward(self, originalImage, renderedImage, predicted, all_classes, min_psnr=30, max_psnr=50, mse_weight = -.00005, psnr_weight = .5):

        returned_reward = 0
        imageLoss = self.calc_mse(originalImage, renderedImage) * mse_weight

        if predicted[1] == self.target:
            # reward = target class confidence * 10 + number of images with target class prediction * .2 - number of images not target class * 1
            returned_reward = predicted[2] * 150 - (self.Num_Imgs - predicted[0]) * .2
        # Reward the agent for predicting anything that is not the true class.
        elif predicted[1] != self.true_class:
            # Rewards the agent for predicting the true class in less images and penalize agent slightly for remaining prediction in true class.
            returned_reward =  (self.Num_Imgs - predicted[0]) * .1 # - predicted[0] * .1
            
            # Reward agent if even 1 of the images were classified as target class.
            for key_class, value in all_classes.items():
                if key_class == self.target:
                    returned_reward += value[0] * .1
                    break
        
        # Penalize agent for significantly altering the image.
        returned_reward += imageLoss

        print("Step Reward: " + str(returned_reward) + " for " + predicted[1] + " with confidence: " + str(predicted[2]) + " and num: " + str((predicted[0])))
        return returned_reward

    def pred_labels(self):
        pred_classes = {}

        preds_per_imgs, imgs = self.mobilenetv2.predict(self.images_output_path, self.Num_Imgs, self.labels, top=1)

        for i in range(len(preds_per_imgs)):
            pred = preds_per_imgs[i][0]
            if pred[1] not in pred_classes:
                pred_classes[pred[1]] = [pred[2], 1]
            else:
                pred_classes[pred[1]][0] *= pred_classes[pred[1]][1]
                pred_classes[pred[1]][0] += pred[2]
                pred_classes[pred[1]][1] += 1
                pred_classes[pred[1]][0] /= pred_classes[pred[1]][1]

        maxKey = str(max(pred_classes, key=lambda x:pred_classes[x][1]))

        print("dictionary: " + str(pred_classes))
        print("pred: " + maxKey + ", confidence: " + str(pred_classes[maxKey][0]) + ", " + str(pred_classes[maxKey][1]) + " images.")

        reward = self.get_reward(self.ground_truth_img, imgs, [pred_classes[maxKey][1], maxKey, pred_classes[maxKey][0]], pred_classes)  # Implement reward function


        info = {
            "Target" : self.target,
            "Predicted" : maxKey,
            "Num_Imgs" : pred_classes[maxKey][1],
            "Confidence" : pred_classes[maxKey][0],
            "Epochs" : self.epochs
        }
        return imgs, maxKey, pred_classes, reward, info

    def step(self, action):
        # Modify feature grid based on action
        self.feature_grid +=  action # Apply action to feature grid

        # Render image from modified feature grid
        self.nerf_model.SaveParameters(self.data_loaded, self.feature_grid)
        self.nerf_model.render_outputs(self.output_file, self.transforms_path, self.images_output_path, width=self.ImageHeight, height=self.ImageWidth)

        truncated, info = False, {}

        images, maxKey, pred_classes, reward, info = self.pred_labels()

        self.total_reward += reward

        # Determine if episode is done
        done = False

        # Setting the "done" flag is for the ".learn". Tells the program when it should stop.
        if (maxKey == self.target and pred_classes[maxKey][0] > .5) or (self.total_reward < -5):
            done = True
            self.total_reward = 0

            print("RESETTING ENVIRONMENT!!!")

            # TODO: Remove after you test to see how reseting the environment after the done flag is set impacts performance.
            self.reset()

        self.epochs += 1

        print("info: " + str(info))
        print("Total Reward: " + str(self.total_reward))

        return images, reward, done, truncated, info

    def close(self):
        self.log.close()