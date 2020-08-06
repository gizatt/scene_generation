#!/usr/bin/python


import os
import dense_correspondence_manipulation.utils.utils as utils
import logging
utils.add_dense_correspondence_to_python_path()
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random
import scipy.stats as ss
import itertools

import torch
from torch.autograd import Variable
from torchvision import transforms


from dense_correspondence_manipulation.utils.constants import *
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import dense_correspondence_manipulation.utils.visualization as vis_utils

import dense_correspondence.evaluation.plotting as dc_plotting
from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image

from synthetic_correspondence_dataset import SyntheticCorrespondenceDataset

''' Vastly cut down from pytorch_dense_correspondence's evaluation.py, which
was dataset-specific and included lots of extra functionality. '''

class DenseCorrespondenceEvaluation(object):
    """
    Helper to test trained dense descriptor networks.
    """

    def __init__(self, config, dataset):
        self._config = config
        self._dataset = dataset

    def load_network_from_config(self, name):
        """
        Loads a network from config file. Puts it in eval mode by default
        :param name:
        :type name:
        :return: DenseCorrespondenceNetwork
        :rtype:
        """
        if name not in self._config["networks"]:
            raise ValueError("Network %s is not in config file" %(name))


        path_to_network_params = self._config["networks"][name]["path_to_network_params"]
        model_folder = os.path.dirname(path_to_network_params)
        dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
        dcn.eval()
        return dcn

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def get_output_dir(self):
        return self._config['output_dir']

    def evaluate_a_random_image(self, dcn):
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self._dataset.get_random_rgbd_make_pose()
        rgb_a_tensor = self._dataset.rgb_image_to_tensor(image_a_rgb)
        res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        res_a = dc_plotting.normalize_descriptor(res_a)
        return image_a_rgb, image_a_depth, image_a_mask, res_a

if __name__ == "__main__":
    data_config_filename = "dataset_config.yaml"
    data_config = utils.getDictFromYamlFilename(data_config_filename)
    dataset = SyntheticCorrespondenceDataset(data_config, debug=False)

    eval_config_filename = "evaluation_config.yaml"
    eval_config = utils.getDictFromYamlFilename(eval_config_filename)
    
    dce = DenseCorrespondenceEvaluation(config=eval_config,
                                        dataset=dataset)
    dcn = dce.load_network_from_config("test_run_2500")

    for k in range(10):
        plt.figure(dpi=300).set_size_inches(4, 4)
        rgb, _, mask, res = dce.evaluate_a_random_image(dcn)
        plt.subplot(2, 1, 1)
        plt.imshow(rgb)
        plt.subplot(2, 1, 2)
        plt.imshow(res)
        plt.show()
