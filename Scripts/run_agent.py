from stable_baselines3 import PPO
from AdversarialEnvironment import InstantNGPEnv
from NeRF import NeRF_Model

import ClipClassifier
import requests
import json
import copy
import numpy as np

pretrained_model = ClipClassifier.CLIP_Classifier()

# Download the ImageNet labels file
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(url)
labels = response.json()

og_file = "../SavedModel.msgpack"
output_file = "../TestBanana.msgpack"
true_class = "banana"
target_class = "slug"
path_to_og_images = "../BananaScene/Output/"
transforms_path =  "../BananaScene/transforms.json"
short_transforms_path = "../BananaScene/short_transforms.json"
images_output_path = "../BananaScene/AdvOut/"
num_imgs = 20


def generate_transforms(og_transforms_path, output_transforms_path, number_of_transforms= 10):
    file = open(og_transforms_path)
    transforms = json.load(file)
    copyJson = copy.deepcopy(transforms)
    with open(output_transforms_path, "w") as short_output:
        copyJson['frames'] = list(np.array(copyJson['frames'])[:number_of_transforms])
        json.dump(copyJson, short_output)

generate_transforms(transforms_path, short_transforms_path, number_of_transforms=31)

nerf_model = NeRF_Model(og_file, output_file, path_to_og_images, short_transforms_path, images_output_path)

# Create environment instance
env = InstantNGPEnv(pretrained_model, nerf_model, true_class, target_class, num_imgs,(800,800), path_to_og_images, images_output_path, output_file, short_transforms_path, labels)

print("GOT TO PPO!")
# Create RL agent
model = PPO('MlpPolicy', env, device='cpu', n_steps=2, batch_size=2, max_grad_norm = .00001)

print("GOT AFTER PPO!")
# Train the agent
# Evaluate the trained agent
episode_reward = 0
obs, info = env.reset()
_states = None

max_num_imgs = 0

epochs = 0
while True:
    model.learn(total_timesteps=1)
    action, _states = model.predict(obs, _states)

    nerf_model.render_outputs(output_file, transforms_path, images_output_path, width=env.ImageHeight, height=env.ImageWidth)

    _, _, _, reward, info = env.pred_labels()

    episode_reward += reward
    print("Current episode reward: " + str(episode_reward))

    # Be careful where you put this if block. If you put it after the "reset environment" if block, it might throw an error, because "reset" returns an empty dictionary.
    if (info['Target'] == info['Predicted'] and info['Confidence'] >= .65 and info["Num_Imgs"] >= 15):
        print("Episode reward:", episode_reward)
        break