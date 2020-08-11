# system
import numpy as np
import os
import fnmatch
import gc
import logging
import time
import shutil
import subprocess
import copy
import yaml

from torch.utils.tensorboard import SummaryWriter

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from scene_generation.inverse_graphics.supervised_dense_descriptors.labeled_descriptors_dataset import LabeledDescriptorsDataset

# Import torch *after* dense correspondence so we load the same version they do
# torch
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


'''
Forked from DenseCorrespondenceTraining in pytorch-dense-correspondence
to separate dependence on SpartanDataset (and switch to my personal synthetic
dataset format).
'''


class DenseCorrespondenceTrainingForSceneGeneration():

    def __init__(self, dataset, dataset_test=None, config=None):
        if config is None:
            config = DenseCorrespondenceTrainingForSceneGeneration.load_default_config()

        self._config = config
        self._dataset = dataset
        self._dataset_test = dataset_test

        self._dcn = None
        self._optimizer = None

    def setup(self):
        """
        Initializes the object
        :return:
        :rtype:
        """
        self.load_dataset()
        self.setup_logging_dir()
        self.setup_tensorboard()


    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def load_dataset(self):
        """
        Loads a dataset, construct a trainloader.
        Additionally creates a dataset and DataLoader for the test data
        :return:
        :rtype:
        """

        batch_size = self._config['training']['batch_size']
        num_workers = self._config['training']['num_workers']
        self._data_loader = torch.utils.data.DataLoader(self._dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

        # create a test dataset
        if self._config["training"]["compute_test_loss"]:
            if self._dataset_test is None:
                self._dataset_test = LabeledDescriptorsDataset(config=self._dataset.config, mode="test")
            self._data_loader_test = torch.utils.data.DataLoader(self._dataset_test, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)

    def build_network(self):
        """
        Builds the DenseCorrespondenceNetwork
        :return:
        :rtype: DenseCorrespondenceNetwork
        """
        return DenseCorrespondenceNetwork.from_config(self._config['dense_correspondence_network'],
                                                      load_stored_params=False)

    def _construct_optimizer(self, parameters):
        """
        Constructs the optimizer
        :param parameters: Parameters to adjust in the optimizer
        :type parameters:
        :return: Adam Optimizer with params from the config
        :rtype: torch.optim
        """
        learning_rate = float(self._config['training']['learning_rate'])
        weight_decay = float(self._config['training']['weight_decay'])
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _get_current_loss(self, logging_dict):
        """
        Gets the current loss for both test and train
        :return:
        :rtype: dict
        """
        d = dict()
        d['train'] = dict()
        d['test'] = dict()
        for key, val in d.items():
            for field in list(logging_dict[key].keys()):
                vec = logging_dict[key][field]
                if len(vec) > 0:
                    val[field] = vec[-1]
                else:
                    val[field] = -1 # placeholder
        return d

    def load_pretrained(self, model_folder, iteration=None):
        """
        Loads network and optimizer parameters from a previous training run.

        Note: It is up to the user to ensure that the model parameters match.
        e.g. width, height, descriptor dimension etc.

        :param model_folder: location of the folder containing the param files 001000.pth. Can be absolute or relative path. If relative then it is relative to pdc/trained_models/
        :type model_folder:
        :param iteration: which index to use, e.g. 3500, if None it loads the latest one
        :type iteration:
        :return: iteration
        :rtype:
        """
        if not os.path.isdir(model_folder):
            pdc_path = utils.getPdcPath()
            model_folder = os.path.join(pdc_path, "trained_models", model_folder)

        # find idx.pth and idx.pth.opt files
        if iteration is None:
            files = os.listdir(model_folder)
            model_param_file = sorted(fnmatch.filter(files, '*.pth'))[-1]
            iteration = int(model_param_file.split(".")[0])
            optim_param_file = sorted(fnmatch.filter(files, '*.pth.opt'))[-1]
        else:
            prefix = utils.getPaddedString(iteration, width=6)
            model_param_file = prefix + ".pth"
            optim_param_file = prefix + ".pth.opt"

        print("model_param_file", model_param_file)
        model_param_file = os.path.join(model_folder, model_param_file)
        optim_param_file = os.path.join(model_folder, optim_param_file)


        self._dcn = self.build_network()
        self._dcn.load_state_dict(torch.load(model_param_file))
        self._dcn.cuda()
        self._dcn.train()

        self._optimizer = self._construct_optimizer(self._dcn.parameters())
        self._optimizer.load_state_dict(torch.load(optim_param_file))

        return iteration

    def run_from_pretrained(self, model_folder, iteration=None, learning_rate=None):
        """
        Wrapper for load_pretrained(), then run()
        """
        iteration = self.load_pretrained(model_folder, iteration)
        if iteration is None:
            iteration = 0

        if learning_rate is not None:
            self._config["training"]["learning_rate_starting_from_pretrained"] = learning_rate
            self.set_learning_rate(self._optimizer, learning_rate)

        self.run(loss_current_iteration=iteration, use_pretrained=True)

    def run(self, loss_current_iteration=0, use_pretrained=False):
        """
        Runs the training
        :return:
        :rtype:
        """

        start_iteration = copy.copy(loss_current_iteration)

        self.setup()
        self.save_configs()

        if not use_pretrained:
            # create new network and optimizer
            self._dcn = self.build_network()
            self._optimizer = self._construct_optimizer(self._dcn.parameters())
        else:
            logging.info("using pretrained model")
            if (self._dcn is None):
                raise ValueError("you must set self._dcn if use_pretrained=True")
            if (self._optimizer is None):
                raise ValueError("you must set self._optimizer if use_pretrained=True")

        # make sure network is using cuda and is in train mode
        dcn = self._dcn
        dcn.cuda()
        dcn.train()

        optimizer = self._optimizer
        batch_size = self._data_loader.batch_size

        loss = 0.

        max_num_iterations = self._config['training']['num_iterations'] + start_iteration
        logging_rate = self._config['training']['logging_rate']
        save_rate = self._config['training']['save_rate']
        compute_test_loss_rate = self._config['training']['compute_test_loss_rate']

        # logging
        self._logging_dict = dict()
        self._logging_dict['train'] = {"iteration": [], "loss": [],
                                       "learning_rate": []}

        self._logging_dict['test'] = {"iteration": [], "loss": []}

        # save network before starting
        if not use_pretrained:
            self.save_network(dcn, optimizer, 0)

        for epoch in range(50):  # loop over the dataset multiple times
            for i, data in enumerate(self._data_loader, 0):
                loss_current_iteration += 1
                start_iter = time.time()

                rgb, target_descriptor = data
                assert(len(rgb.shape) == 4)
                assert(len(target_descriptor.shape) == 4)
                rgb_tensor = self.dataset.rgb_image_to_tensor(np.asarray(rgb).copy(), normalize=True)
                # Assume batched
                target_descriptor_tensor = torch.tensor(np.asarray(target_descriptor).astype(np.float)).permute([0, 3, 1, 2]) / 255. # Recale range to 0, 1

                rgb_tensor = Variable(rgb_tensor.cuda(), requires_grad=False)
                target_descriptor_tensor = Variable(target_descriptor_tensor.cuda(), requires_grad=False)

                optimizer.zero_grad()
                self.adjust_learning_rate(optimizer, loss_current_iteration)

                # run both images through the network
                pred_descriptor = dcn.forward(rgb_tensor)

                # get loss
                loss = nn.MSELoss()(target_descriptor_tensor, pred_descriptor)

                loss.backward()
                optimizer.step()

                print("%04d loss: %f" % (loss_current_iteration, loss.item()))


                elapsed = time.time() - start_iter

                def update_plots():
                    """
                    Updates the tensorboard plots with current loss function information
                    :return:
                    :rtype:
                    """

                    learning_rate = DenseCorrespondenceTrainingForSceneGeneration.get_learning_rate(optimizer)
                    self._logging_dict['train']['learning_rate'].append(learning_rate)
                    self._tensorboard_writer.add_scalar("learning_rate", learning_rate, loss_current_iteration)

                    # loss is never zero
                    self._tensorboard_writer.add_scalar("train_loss", loss.item(), loss_current_iteration)

                    if (loss_current_iteration % 50 == 1):
                        self._tensorboard_writer.add_images("rgb", torch.tensor(rgb), loss_current_iteration, dataformats="NHWC")
                        self._tensorboard_writer.add_images("target", target_descriptor_tensor, loss_current_iteration, dataformats="NCHW")
                        self._tensorboard_writer.add_images("predicted", torch.clamp(pred_descriptor, 0., 1.), loss_current_iteration, dataformats="NCHW")

                update_plots()

                if loss_current_iteration % save_rate == 0:
                    self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)

                if loss_current_iteration % logging_rate == 0:
                    logging.info("Training on iteration %d of %d" %(loss_current_iteration, max_num_iterations))

                    logging.info("single iteration took %.3f seconds" %(elapsed))

                    percent_complete = loss_current_iteration * 100.0/(max_num_iterations - start_iteration)
                    logging.info("Training is %d percent complete\n" %(percent_complete))

                # don't compute the test loss on the first few times through the loop
                if self._config["training"]["compute_test_loss"] and (loss_current_iteration % compute_test_loss_rate == 0) and loss_current_iteration > 5:
                    logging.info("Computing test loss")

                    # delete the loss, match_loss, non_match_loss variables so that
                    # pytorch can use that GPU memory
                    del loss
                    gc.collect()

                    dcn.eval()
                    raise NotImplementedError()
                    test_loss, test_match_loss, test_non_match_loss = DCE.compute_loss_on_dataset(
                        dcn,
                        self._data_loader_test, self._config['loss_function'], num_iterations=self._config['training']['test_loss_num_iterations'])

                    # delete these variables so we can free GPU memory
                    del test_loss, test_match_loss, test_non_match_loss

                    # make sure to set the network back to train mode
                    dcn.train()

                if loss_current_iteration % self._config['training']['garbage_collect_rate'] == 0:
                    logging.debug("running garbage collection")
                    gc_start = time.time()
                    gc.collect()
                    gc_elapsed = time.time() - gc_start
                    logging.debug("garbage collection took %.2d seconds" %(gc_elapsed))

                if loss_current_iteration > max_num_iterations:
                    logging.info("Finished testing after %d iterations" % (max_num_iterations))
                    self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)
                    return


    def setup_logging_dir(self):
        """
        Sets up the directory where logs will be stored and config
        files written
        :return: full path of logging dir
        :rtype: str
        """

        if 'logging_dir_name' in self._config['training']:
            dir_name = self._config['training']['logging_dir_name']
        else:
            dir_name = utils.get_current_time_unique_name() +"_" + str(self._config['dense_correspondence_network']['descriptor_dimension']) + "d"

        self._logging_dir_name = dir_name

        self._logging_dir = os.path.join(self._config['training']['logging_dir'], dir_name)

        print("logging_dir:", self._logging_dir)

        if os.path.isdir(self._logging_dir):
            shutil.rmtree(self._logging_dir)

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

        # make the tensorboard log directory
        self._tensorboard_log_dir = os.path.join(self._logging_dir, "tensorboard")
        if not os.path.isdir(self._tensorboard_log_dir):
            os.makedirs(self._tensorboard_log_dir)

        return self._logging_dir

    @property
    def logging_dir(self):
        """
        Sets up the directory where logs will be stored and config
        files written
        :return: full path of logging dir
        :rtype: str
        """
        return self._logging_dir

    def save_network(self, dcn, optimizer, iteration, logging_dict=None):
        """
        Saves network parameters to logging directory
        :return:
        :rtype: None
        """

        network_param_file = os.path.join(self._logging_dir, utils.getPaddedString(iteration, width=6) + ".pth")
        optimizer_param_file = network_param_file + ".opt"
        torch.save(dcn.state_dict(), network_param_file)
        torch.save(optimizer.state_dict(), optimizer_param_file)

        # also save loss history stuff
        if logging_dict is not None:
            log_history_file = os.path.join(self._logging_dir, utils.getPaddedString(iteration, width=6) + "_log_history.yaml")
            utils.saveToYaml(logging_dict, log_history_file)

            current_loss_file = os.path.join(self._logging_dir, 'loss.yaml')
            current_loss_data = self._get_current_loss(logging_dict)

            utils.saveToYaml(current_loss_data, current_loss_file)



    def save_configs(self):
        """
        Saves config files to the logging directory
        :return:
        :rtype: None
        """
        training_params_file = os.path.join(self._logging_dir, 'training.yaml')
        utils.saveToYaml(self._config, training_params_file)

        dataset_params_file = os.path.join(self._logging_dir, 'dataset.yaml')
        utils.saveToYaml(self._dataset.config, dataset_params_file)

        # make unique identifier
        identifier_file = os.path.join(self._logging_dir, 'identifier.yaml')
        identifier_dict = dict()
        identifier_dict['id'] = utils.get_unique_string()
        utils.saveToYaml(identifier_dict, identifier_file)


    def adjust_learning_rate(self, optimizer, iteration):
        """
        Adjusts the learning rate according to the schedule
        :param optimizer:
        :type optimizer:
        :param iteration:
        :type iteration:
        :return:
        :rtype:
        """

        steps_between_learning_rate_decay = self._config['training']['steps_between_learning_rate_decay']
        if iteration % steps_between_learning_rate_decay == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self._config["training"]["learning_rate_decay"]

    @staticmethod
    def set_learning_rate(optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    @staticmethod
    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        return lr

    def setup_tensorboard(self):
        """
        Starts the tensorboard server and sets up the plotting
        :return:
        :rtype:
        """

        # start 
        # cmd = "python -m tensorboard.main"
        logging.info("setting up tensorboard_logger")
        cmd = "tensorboard --logdir=%s" %(self._tensorboard_log_dir)
        self._tensorboard_writer = SummaryWriter(log_dir=self._tensorboard_log_dir)
        logging.info("tensorboard logger started")

    @staticmethod
    def load_default_config():
        config = utils.getDictFromYamlFilename("training_config.yaml")
        return config


if __name__ == "__main__":
    data_config_filename = "dataset_config.yaml"
    with open(data_config_filename, "r") as f:
        data_config = yaml.load(f)
    dataset = LabeledDescriptorsDataset(data_config)

    train_config_file = "training_config.yaml"
    with open(train_config_file, "r") as f:
            train_config = yaml.load(f)
    logging_dir = "trained_models/test_run_updated_torch"
    d = 3 # the descriptor dimension
    name = "prime_box"
    train_config["training"]["logging_dir_name"] = name
    train_config["training"]["logging_dir"] = logging_dir
    train_config["dense_correspondence_network"]["descriptor_dimension"] = d

    train = DenseCorrespondenceTrainingForSceneGeneration(dataset=dataset, config=train_config)
    train.run()