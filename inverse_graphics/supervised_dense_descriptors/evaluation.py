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


from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from labeled_descriptors_dataset import LabeledDescriptorsDataset

''' Vastly cut down from pytorch_dense_correspondence's evaluation.py, which
was dataset-specific and included lots of extra functionality. '''

class SupervisedDescriptorEvaluation(object):
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
        rgb, target_descriptors = self._dataset[np.random.randint(len(self._dataset))]
 
        rgb_tensor = self.dataset.rgb_image_to_tensor(np.asarray(rgb).copy(), normalize=True)
        pred_descriptors = dcn.forward_single_image_tensor(rgb_tensor).data.cpu().numpy()

        #res_a = dc_plotting.normalize_descriptor(res_a)
        return (
            rgb,
            target_descriptors,
            pred_descriptors
        )

if __name__ == "__main__":
    data_config_filename = "dataset_config.yaml"
    data_config = utils.getDictFromYamlFilename(data_config_filename)
    dataset = LabeledDescriptorsDataset(data_config, debug=False)

    eval_config_filename = "evaluation_config.yaml"
    eval_config = utils.getDictFromYamlFilename(eval_config_filename)
    
    sce = SupervisedDescriptorEvaluation(config=eval_config,
                                        dataset=dataset)
    dcn = sce.load_network_from_config("test_run")

    fig = plt.figure(dpi=300)
    fig.set_size_inches(4, 4)
    height = 4
    for k in range(height):
        rgb, target, predicted = sce.evaluate_a_random_image(dcn)
        plt.subplot(height, 3, k*3 + 1)
        if k == 0:
            plt.title("Input")
        plt.imshow(rgb)
        plt.subplot(height, 3, k*3 + 2)
        if k == 0:
            plt.title("Target")
        plt.imshow(target)
        plt.subplot(height, 3, k*3 + 3)
        if k == 0:
            plt.title("Predicted")
        plt.imshow(predicted)
    plt.show()
