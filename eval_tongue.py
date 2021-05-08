import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import skimage.io
from PIL import Image
from mrcnnn.config import Config
from mrcnnn import utils
from mrcnnn import model as modellib, utils
import mrcnnn.model as modellib
from mrcnnn import visualize
from mrcnnn.model import log


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Configurations


class ShapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    # anchor side in pixels
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)


config = ShapesConfig()
config.display()


class ShapesDataset(utils.Dataset):
    # 得到該圖有多少物件
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # labelme中得到的yaml文件，從而得到mask每一層的標籤
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新寫draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新寫load_shapes，包含自己的類別
    # 在self.image_info中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        # Add classes
        self.add_class("shapes", 1, "tongue")

        for i in range(count):
            # 得圖片寬跟高
            print(i)
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" +
                  filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path +
                                "labelme_json/" + filestr + "_json/img.png")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重寫load_mask
    def load_mask(self, image_id):
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros(
            [info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))

        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []

        for i in range(len(labels)):
            if labels[i].find("tongue") != -1:
                labels_form.append("tongue")
            """
            elif labels[i].find("leg") != -1:
                 print "leg"
                labels_form.append("leg")
            """
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def merged_mask(masks):
    """
    merge mask into one and return merged mask
    """
    n = masks.shape[2]

    if n != 0:
        merged_mask = np.zeros((masks.shape[0], masks.shape[1]))
        for i in range(n):
            merged_mask += masks[..., i]
        merged_mask = np.asarray(merged_mask, dtype=np.uint8)
        return merged_mask
    return masks[:, :, 0]


def compute_iou(predict_mask, gt_mask):
    """
    Computes Intersection over Union score for two binary masks.
    :param predict_mask: numpy array
    :param gt_mask: numpy array
    :type1 and type2 results are same
    :return iou score:
    """
    if predict_mask.shape[2] == 0:
        return 0
    mask1 = merged_mask(predict_mask)
    mask2 = merged_mask(gt_mask)

    # type 1
    """
    intersection = np.sum((mask1 + mask2) > 1)
    union = np.sum((mask1 + mask2) > 0)
    iou_score = intersection / float(union)
    #print("Iou 1 : ",iou_score)
    """
    # type2
    intersection = np.logical_and(mask1, mask2)  
    union = np.logical_or(mask1, mask2) 
    iou_score = np.sum(intersection) / np.sum(union)
    print("Iou : ", iou_score)
    return iou_score


dataset_root_path = "./tongue_eval/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv_mask"
imglist = os.listdir(img_floder)
count = len(imglist)

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(30, img_floder, mask_floder,
                        imglist, dataset_root_path)
dataset_val.prepare()


# Recreate the model in inference mode
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)


# Load trained weights
model_path = os.path.join(ROOT_DIR, "mask_rcnn_shapes_0080.h5")
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# 暫時的 要先跑一個下面才可以跑  系統問題
image_tmp = skimage.io.imread("./TestImage/tongue_1.jpg")
results_tmp = model.detect([image_tmp], verbose=0)


# Evaluation
image_ids = np.random.choice(dataset_val.image_ids, 10)
IOUs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                        image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]

    # Compute IOU
    IOU = compute_iou(r['masks'], gt_mask)
    IOUs.append(IOU)

print("mIOU: ", np.mean(IOUs))
print("standard deviation : ", np.std(IOUs, ddof=1))
