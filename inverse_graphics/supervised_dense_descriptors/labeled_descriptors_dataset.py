import os
import numpy as np
import logging
import glob
import random
import copy
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from yaml import CLoader as Loader

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import dense_correspondence_manipulation.utils.constants as constants
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation

'''
Collates the raw RGB + labeled descriptor images from
my synthetic datset format.
'''


class LabeledDescriptorsDataset(Dataset):

    def __init__(self, config, mode="train", verbose=False, debug=False, augmentation=True):
        """
        :param config: This is for creating a dataset from a dataset config file.

        :type config: dict()
        """

        self._verbose = verbose
        self._debug = debug
        self._augmentation = augmentation

        # Parse and load in config values
        self._collect_scene_data(config)
        self._initialize_rgb_image_to_tensor()

        self.mode = mode
        assert mode in ["train", "test"], mode

        print("Using SyntheticCorrespondenceDataset:")
        print("   - in", self.mode, "mode")
        print("   - total images:    ", len(self._all_image_paths[self.mode]))

    def __getitem__(self, index):
        """
        """
        rgb_image_path = self._all_image_paths[self.mode][index]
        descriptor_image_path = rgb_image_path[:-4] + "_descriptors.png"

        rgb = np.asarray(Image.open(rgb_image_path).convert('RGB'))
        descriptor = np.asarray(Image.open(descriptor_image_path).convert('RGB'))

        if self._augmentation:
            mask = 1. - np.all(descriptor == 0., axis=-1)
            rgb = np.asarray(correspondence_augmentation.random_domain_randomize_background(rgb, mask))

            # Sometimes do flipping
            if random.random() >= 0.5:
                rgb = np.flip(rgb, axis=[0, 1]).copy()
                descriptor = np.flip(descriptor, axis=[0, 1]).copy()

        if self._debug:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(rgb)
            plt.subplot(2, 1, 2)
            plt.imshow(descriptor)
            plt.show()

        return rgb, descriptor

    def __len__(self):
        return len(self._all_image_paths[self.mode])

    def _collect_scene_data(self, config):
        """
        Scans all the scenes in the dataset and builds a list
        of all of the base rgb image paths.
        """

        self._config = config
        self.scenes_root_path = config['scenes_root_path']
        assert(os.path.isdir(self.scenes_root_path))

        self._scene_dict = dict()
        # each one is a list of scenes
        self._all_image_paths = {"train": [], "test": []}

        for key, val in self._all_image_paths.items():
            for scene_collection_name in config[key]:
                scene_collection_dir = os.path.join(self.scenes_root_path, scene_collection_name)
                assert os.path.isdir(scene_collection_dir), scene_collection_dir
                # Scan all scenes in this scene dir
                for scene_name in os.listdir(scene_collection_dir):
                    full = os.path.join(scene_collection_dir, scene_name)
                    if os.path.isdir(full):
                        val += self._get_all_rgb_image_paths_in_scene_dir(full)

    def _get_all_rgb_image_paths_in_scene_dir(self, scene_path):
        # Laziest way that works right now -- it's just all the jpgs.
        # (All other labels + info images are pngs.)
        return glob.glob(scene_path + "/*.jpg")


    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor_unnormalized = transforms.ToTensor()
        self._rgb_image_to_tensor_normalized = transforms.Compose([transforms.ToTensor(), norm_transform])

    def get_image_mean(self):
        """
        Returns dataset image_mean
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_MEAN

        # return self.config["image_normalization"]["mean"]


        return constants.DEFAULT_IMAGE_MEAN

    def get_image_std_dev(self):
        """
        Returns dataset image std_dev
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_STD_DEV

        # return self.config["image_normalization"]["std_dev"]

        return constants.DEFAULT_IMAGE_STD_DEV

    def rgb_image_to_tensor(self, img, normalize=True):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """
        if normalize:
            func = self._rgb_image_to_tensor_normalized
        else:
            func = self._rgb_image_to_tensor_unnormalized
        if len(img.shape) == 4:
            # torchvision ToTensor / Normalize do not like batch operations :()
            return torch.stack([func(img[k, :, :, :].copy()) for k in range(img.shape[0])], dim=0)
        else:
            return func(img.copy())

    @property
    def config(self):
        return self._config

if __name__ == "__main__":
    data_config_filename = "dataset_config.yaml"
    with open(data_config_filename, "r") as f:
        data_config = yaml.load(f, Loader=Loader)
    dataset = LabeledDescriptorsDataset(data_config, debug=True)
    for k in range(10):
        dataset[k]
    print("Loaded from dataset happily.")