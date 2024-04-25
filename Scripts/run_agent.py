from stable_baselines3 import PPO
from AdversarialEnvironment import InstantNGPEnv
from NeRF import NeRF_Model

import ClipClassifier
import requests
import json
import copy
import numpy as np
import argparse


def main(args):
    print("temp")
    pretrained_model = ClipClassifier.CLIP_Classifier(args.clip_device)

    # Download the ImageNet labels file
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    labels = response.json()
    def generate_transforms(og_transforms_path, output_transforms_path, number_of_transforms= 10):
        file = open(og_transforms_path)
        transforms = json.load(file)
        copyJson = copy.deepcopy(transforms)
        with open(output_transforms_path, "w") as short_output:
            copyJson['frames'] = list(np.array(copyJson['frames'])[:number_of_transforms])
            json.dump(copyJson, short_output)

    generate_transforms(args.transforms_path, args.output_transforms_path, number_of_transforms=args.num_imgs)

    nerf_model = NeRF_Model(args.input_saved_nerf_file_path, args.output_saved_nerf_file_path, args.input_image_folder, args.output_transforms_path, args.output_image_folder)

    # Create environment instance
    env = InstantNGPEnv(pretrained_model, nerf_model, args.true_class, args.target_class, args.num_imgs,(args.image_width,args.image_height), args.input_image_folder, args.output_image_folder, args.output_saved_nerf_file_path, args.output_transforms_path, labels)

    # Create RL agent
    model = PPO('MlpPolicy', env, device='cpu', n_steps=2, batch_size=2, max_grad_norm = .00001)
    # Train the agent
    # Evaluate the trained agent
    episode_reward = 0
    obs, info = env.reset()
    _states = None

    while True:
        model.learn(total_timesteps=1)
        action, _states = model.predict(obs, _states)

        nerf_model.render_outputs(args.output_saved_nerf_file_path, args.output_transforms_path, args.output_image_folder, width=args.image_width, height=args.image_height)

        _, _, _, reward, info = env.pred_labels()

        episode_reward += reward
        print("Current episode reward: " + str(episode_reward))

        # Be careful where you put this if block. If you put it after the "reset environment" if block, it might throw an error, because "reset" returns an empty dictionary.
        if (info['Target'] == info['Predicted'] and info['Confidence'] >= .65 and info["Num_Imgs"] >= 15):
            print("Episode reward:", episode_reward)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement learning approach for generating adversarial noise.")
    parser.add_argument("--clip_device", default="cuda:0", type=str, help="Which device to execute clip on.")
    parser.add_argument("--input_image_folder", default="../BananaScene/Output/", type=str, help="Path to the images we will use to generate a 3D model using NeRF.")
    parser.add_argument("--output_image_folder", default="../BananaScene/AdvOut/", type=str, help="Folder to output generated images to.")
    parser.add_argument("--image_width", default=800, type=int, help="Width of rendered image from NeRF.")
    parser.add_argument("--image_height", default=800, type=int, help="Width of rendered image from NeRF.")
    parser.add_argument("--transforms_path", default="../BananaScene/transforms.json", type=str, help="Path to the original JSON path")
    parser.add_argument("--output_transforms_path", default="../BananaScene/short_transforms.json", type=str, help="Output path we will store the modified transforms file.")
    parser.add_argument("--input_saved_nerf_file_path", default="../SavedModel.msgpack", type=str, help="Path to our input saved NeRF model file.")
    parser.add_argument("--output_saved_nerf_file_path", default="../TestBanana.msgpack", type=str, help="Path to where we will save our generated NeRF model file.")

    parser.add_argument("--true_class", default="banana", type=str, help="True class of the object.")
    parser.add_argument("--target_class", default="slug", type=str, help="Target class to produce adversarial noise to target.")
    parser.add_argument("--num_imgs", default=20, type=int, help="Number of images to use as observation space for reinforcement learning.")

    main(parser)