#!/usr/bin/python


import os
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.stats as ss
import sys
import itertools
from PIL import Image

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()

import torch
from torch.autograd import Variable
from torchvision import transforms

from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from scene_generation.inverse_graphics.supervised_dense_descriptors.labeled_descriptors_dataset import LabeledDescriptorsDataset

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

    def evaluate_image(self, dcn, rgb):
        rgb_tensor = self.dataset.rgb_image_to_tensor(np.asarray(rgb).copy(), normalize=True)
        return dcn.forward_single_image_tensor(rgb_tensor).data.cpu().numpy()

if __name__ == "__main__":
    data_config_filename = "dataset_config.yaml"
    data_config = utils.getDictFromYamlFilename(data_config_filename)
    dataset = LabeledDescriptorsDataset(data_config, debug=False, augmentation=False)

    eval_config_filename = "evaluation_config.yaml"
    eval_config = utils.getDictFromYamlFilename(eval_config_filename)
    
    sce = SupervisedDescriptorEvaluation(config=eval_config,
                                        dataset=dataset)
    dcn = sce.load_network_from_config("test_run_updated_torch")

    # Show a couple generated descriptor maps
    if (1):
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

        # Descriptor maps on real images
        plt.figure(dpi=300)
        fig.set_size_inches(8, 2)
        ims_dir = "/home/gizatt/data/generated_cardboard_envs/real_prime_box_images"
        ims = os.listdir(ims_dir)
        for k, im in enumerate(ims):
            plt.subplot(len(ims), 2, k*2 + 1)
            rgb = Image.open(os.path.join(ims_dir, im))
            w, h = rgb.width, rgb.height
            factor = 640. / w
            rgb = rgb.resize((int(factor * w), int(factor * h)))
            plt.imshow(rgb)
            plt.subplot(len(ims), 2, k*2 + 2)
            out = sce.evaluate_image(dcn, rgb)
            plt.imshow(out)
            Image.fromarray((np.clip(out, 0., 1.)*255).astype(np.uint8)).save(im + "_pred.png")

    sys.exit(0)
    # Show a pair of side-by-side descripts + interactively highlight
    fig = plt.figure(dpi=300)
    fig.set_size_inches(4, 4)
    rgb_a, _, res_a = sce.evaluate_a_random_image(dcn)
    rgb_b, _, res_b = sce.evaluate_a_random_image(dcn)
    ax_to_mouseover = plt.subplot(2, 2, 1)
    rgb_a = np.asarray(rgb_a).astype(float) / 255.
    rgb_a_im = plt.imshow(rgb_a)
    plt.subplot(2, 2, 2)
    plt.imshow(res_a)

    rgb_b = np.asarray(rgb_b).astype(float) / 255.
    ax_to_update = plt.subplot(2, 2, 3)
    rgb_b_im = plt.imshow(rgb_b)
    plt.subplot(2, 2, 4)
    plt.imshow(res_b)

    def hover(event):
        if ax_to_mouseover.contains(event)[0]:
            u = int(event.xdata)
            v = int(event.ydata)
            descriptor = res_a[v, u]
            # Distance from First descriptor image to this one
            distances = np.linalg.norm(res_a - descriptor, axis=-1, keepdims=True)
            distances = np.tile(distances, (1, 1, 3))
            im_updated = rgb_a * 0.5 + rgb_a * 0.5 * (distances < 0.1)
            rgb_a_im.set_data(im_updated)

            # Distance from second descriptor image to this one
            distances = np.linalg.norm(res_b - descriptor, axis=-1, keepdims=True)
            distances = np.tile(distances, (1, 1, 3))
            im_updated = rgb_b * 0.5 + rgb_b * 0.5 * (distances < 0.1)
            rgb_b_im.set_data(im_updated)
        else:
            rgb_a_im.set_data(rgb_a)
            rgb_b_im.set_data(rgb_b)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)           
    plt.show()

