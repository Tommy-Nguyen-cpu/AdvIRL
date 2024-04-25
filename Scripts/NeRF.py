import json
import os
import msgpack
import numpy as np
from scenes import *
import pyngp as ngp

class NeRF_Model():
    def __init__(self, og_file, output_file, path_to_og_images, transforms_path, images_output_path):
        self.testbed = ngp.Testbed()
        self.og_file = og_file
        self.output_file = output_file
        self.path_to_og_images = path_to_og_images
        self.transforms_path =  transforms_path
        self.images_output_path = images_output_path
    def get_scene(self, scene):
        for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
            if scene in scenes:
                return scenes[scene]
        return None

    def render_outputs(self, load_snapshot, screenshot_transforms, screenshot_dir, sharpen=0, exposure=0.0, screenshot_frames=None, screenshot_spp=16, width=224, height=224, network=None):
        if load_snapshot:
            scene_info = self.get_scene(load_snapshot)
            if scene_info is not None:
                load_snapshot = default_snapshot_filename(scene_info)
            self.testbed.load_snapshot(load_snapshot)

        ref_transforms = {}
        if screenshot_transforms: # try to load the given file straight away
            print("Screenshot transforms from ", screenshot_transforms)
            with open(screenshot_transforms) as f:
                ref_transforms = json.load(f)

        if self.testbed.mode == ngp.TestbedMode.Sdf:
            self.testbed.tonemap_curve = ngp.TonemapCurve.ACES

        self.testbed.nerf.sharpen = float(sharpen)
        self.testbed.exposure = exposure
        self.testbed.shall_train = True


        self.testbed.nerf.render_with_lens_distortion = True

        if ref_transforms:
            self.testbed.fov_axis = 0
            self.testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
            if not screenshot_frames:
                screenshot_frames = range(len(ref_transforms["frames"]))
            
            self.testbed.background_color = [0.0, 0.0, 0.0, 0.0]
            for idx in screenshot_frames:
                f = ref_transforms["frames"][int(idx)]
                if 'transform_matrix' in f:
                    cam_matrix = f['transform_matrix']
                elif 'transform_matrix_start' in f:
                    cam_matrix = f["transform_matrix_start"]
                else:
                    raise KeyError()
                self.testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
                outname = os.path.join(screenshot_dir, os.path.basename(f["file_path"]))

                # Some NeRF datasets lack the .png suffix in the dataset metadata
                if not os.path.splitext(outname)[1]:
                    outname = outname + ".png"

                image = self.testbed.render(width or int(ref_transforms["w"]), height or int(ref_transforms["h"]), screenshot_spp, True)
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                write_image(outname, image)

    def loadNeRFData(self):
        with open(self.og_file, "rb") as data_file:
            byte_data = data_file.read()
            data_loaded = msgpack.unpackb(byte_data)
            parameters = np.frombuffer(data_loaded["snapshot"]["params_binary"], dtype=np.float16).copy()
            return data_loaded, parameters
            
    def SaveParameters(self, data_loaded, parameters):
        data_loaded['snapshot']['params_binary'] = parameters.tobytes()

        with open(self.output_file, "wb") as file:
            file.write(msgpack.packb(data_loaded))