# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class ImageSegmentation:
  def ModelSetup(self):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return cfg,predictor
    
  def Visualize(self, im_path, config, predictor, scale_param = 1.2):
    image = cv2.imread(im_path)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(config.DATASETS.TRAIN[0]), scale=scale_param)
    out = v.draw_instance_predictions(predictor(image)["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

  # For some odd reason, the keys are shifted by 1? Banana in the class list (https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda) is 47, but the predictor says its 46.
  def RemoveBackground(self, image, predictor, objectClassID, filename):
    output = predictor(image)
    mask = output["instances"].pred_masks.cpu().numpy()
    pred_classes = output['instances'].pred_classes.cpu().numpy()
    indexes_of_true_class = np.where(pred_classes == objectClassID)[0]

    # If Detectron2 doesn't pick up the object.
    if len(indexes_of_true_class) == 0:
      return
    # print("predicted classes: " + str(pred_classes) + " with index for banana " + str(indexes_of_true_class) + " for filename " + filename)

    combined_mask = np.any(mask[indexes_of_true_class], axis=0)

    # Removes background
    image[~combined_mask] = 0

    # Save updated image.
    cv2.imwrite(filename, image)