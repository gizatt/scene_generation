from dense_correspondence.dataset.dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
import dense_correspondence_manipulation.utils.transformations as transformations

import os
import numpy as np
import logging
import glob
import random
import copy
from PIL import Image

import torch

# note that this is the torchvision provided by the warmspringwinds
# pytorch-segmentation-detection repo. It is a fork of pytorch/vision
from torchvision import transforms

import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation
import dense_correspondence_manipulation.utils.constants as constants
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics

'''
Based on SpartanDataset from pytorch_dense_correspondence.

Organized into a set of scenes, which has
simultaneously sets of views into the scene as it evolves through time.
So the total set of indices we'll use are the scene index, the
camera index, and the time index. 

Broadly, the dataset save format is:
- A scene collection folder (like "scene_group_5_types_100")
- Containing a set of scene folders (like "scene_042")
- Containing a set of images and a "scene_info.yaml" folder,
- where each image has a name "X_Y[_drake_label.png, _drake_depth.png, .jpg"]
- where X indicates the camera # and Y indicates the timestep #. Generally,
I've been saving scenes with 5 cameras and a single timestep.

See the dataset overview readme for how scene_info stores info about all
of that.
'''


class SyntheticCorrespondenceDataset(DenseCorrespondenceDataset):

    def __init__(self, config, debug=False, mode="train", verbose=False):
        """
        :param config: This is for creating a dataset from a dataset config file.

        :type config: dict()
        """

        DenseCorrespondenceDataset.__init__(self, debug=debug)

        # Otherwise, all of these parameters should be set in
        # set_parameters_from_training_config()
        # which is called from training.py
        # and parameters are populated in training.yaml
        if self.debug:
            # NOTE: these are not the same as the numbers
            # that get plotted in debug mode.
            # This is just so the dataset will "run".
            self._domain_randomize = False
            self.num_masked_non_matches_per_match = 5
            self.num_background_non_matches_per_match = 5
            self.cross_scene_num_samples = 1000
            self._use_image_b_mask_inv = True
            self.num_matching_attempts = 10000
            self.sample_matches_only_off_mask = True

        self._verbose = verbose

        # Parse and load in config values
        self._setup_scene_data(config)

        self._scene_metadata = dict()
        self._initialize_rgb_image_to_tensor()

        if mode == "test":
            self.set_test_mode()
        elif mode == "train":
            self.set_train_mode()
        else:
            raise ValueError("mode should be one of [test, train]")

        self.init_length()
        print("Using SyntheticCorrespondenceDataset:")
        print("   - in", self.mode, "mode")
        print("   - number of scenes", self._num_scenes)
        print("   - total images:    ", self.num_images_total)

    def set_parameters_from_training_config(self, training_config):
        """
        Much of this is copied from the superclass, but some SpartanDataset-specific
        stuff crept in there.

        Some parameters that are really associated only with training, for example
        those associated with random sampling during the training process,
        should be passed in from a training.yaml config file.

        :param training_config: a dict() holding params
        """

        if (self.mode == "train") and (training_config["training"]["domain_randomize"]):
            logging.info("enabling domain randomization")
            self.enable_domain_randomization()
        else:
            self.disable_domain_randomization()

        # self._training_config = copy.deepcopy(training_config["training"])

        self.num_matching_attempts = int(training_config['training']['num_matching_attempts'])
        self.sample_matches_only_off_mask = training_config['training']['sample_matches_only_off_mask']

        self.num_non_matches_per_match = training_config['training']["num_non_matches_per_match"]


        self.num_masked_non_matches_per_match     = int(training_config['training']["fraction_masked_non_matches"] * self.num_non_matches_per_match)

        self.num_background_non_matches_per_match = int(training_config['training'][
                                                    "fraction_background_non_matches"] * self.num_non_matches_per_match)

        self.cross_scene_num_samples              = training_config['training']["cross_scene_num_samples"]

        self._use_image_b_mask_inv = training_config["training"]["use_image_b_mask_inv"] 

    def __getitem__(self, index):
        """
        This overloads __getitem__ and is what is actually returned
        using a torch dataloader.

        This small function randomly chooses one of our different
        img pair types, then returns that type of data.

        TODO(gizatt) Do we ever need to use the index?
        """
        scene_name = self.get_random_scene_name()
        return self.get_within_scene_data(scene_name)


    def _setup_scene_data(self, config):
        """
        Initializes the data for all the scenes in the dataset.

        self._scene_dict has (key, value) = ("train"/"test", list of scenes)

        Note that the scenes have absolute paths here
        """

        self._config = config
        self.scenes_root_path = config['scenes_root_path']
        assert(os.path.isdir(self.scenes_root_path))

        self._scene_dict = dict()
        # each one is a list of scenes
        self._scene_dict = {"train": [], "test": []}

        for key, val in self._scene_dict.items():
            for scene_collection_name in config[key]:
                scene_collection_dir = os.path.join(self.scenes_root_path, scene_collection_name)
                assert os.path.isdir(scene_collection_dir), scene_collection_dir
                # Scan all scenes in this scene dir
                for scene_name in os.listdir(scene_collection_dir):
                    full = os.path.join(scene_collection_dir, scene_name)
                    if os.path.isdir(full):
                        # Save the scene name as the scene collection
                        # name + the scene name to fully disambiguate
                        val.append(os.path.join(scene_collection_name, scene_name))

    def scene_generator(self, mode=None):
        """
        Returns a generator that traverses all the scenes
        :return:
        :rtype:
        """
        if mode is None:
            mode = self.mode

        for scene_name in self._scene_dict[mode]:
            yield scene_name

    def init_length(self):
        """
        Computes the total number of images and scenes in this dataset.
        Sets the result to the class variables self.num_images_total and self._num_scenes
        :return:
        :rtype:
        """
        self.num_images_total = 0
        self._num_scenes = 0
        for scene_name in self.scene_generator():
            scene_directory = self.get_full_path_for_scene(scene_name)
            # Latch on to the jpgs, of which there are only one per view
            # into the scene (alongside a pile of pngs of depth images
            # and annotations and things).
            rgb_images_regex = os.path.join(scene_directory, "*.jpg")
            all_rgb_images_in_scene = glob.glob(rgb_images_regex)
            num_images_this_scene = len(all_rgb_images_in_scene)
            self.num_images_total += num_images_this_scene
            self._num_scenes += 1

    def get_scene_list(self, mode=None):
        """
        Returns a list of all scenes in this dataset
        :return:
        :rtype:
        """
        scene_generator = self.scene_generator(mode=mode)
        scene_list = []
        for scene_name in scene_generator:
            scene_list.append(scene_name)

        return scene_list
    
    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])

    def get_full_path_for_scene(self, scene_name):
        """
        Returns the full path to the processed logs folder
        :param scene_name:
        :type scene_name:
        :return:
        :rtype:
        """
        return os.path.join(self.scenes_root_path, scene_name)


    def load_all_scene_metadata(self):
        """
        Efficiently pre-loads all pose data for the scenes. This is because when used as
        part of torch DataLoader in threaded way it behaves strangely
        :return:
        :rtype:
        """

        for scene_name in self.scene_generator():
            self.get_scene_metadata(scene_name)

    def get_scene_metadata(self, scene_name):
        """
        Checks if have not already loaded the scene_info.yaml for this scene,
        if haven't then loads it. Then returns the dict of the scene_info.yaml.
        :type scene_name: str
        :return: a dict() of the scene_info.yaml for the scene.
        :rtype: dict()
        """
        # TODO(gizatt) Update
        if scene_name not in self._scene_metadata:
            logging.info("Loading scene data for scene %s" %(scene_name) )
            scene_metadata_filename = os.path.join(self.get_full_path_for_scene(scene_name),
                                              'scene_info.yaml')
            self._scene_metadata[scene_name] = utils.getDictFromYamlFilename(scene_metadata_filename)

        return self._scene_metadata[scene_name]

    def make_image_string_index(self, time_index, cam_index):
        return "%02d_%08d" % (cam_index, time_index)

    def split_image_string_index(self, img_index):
        assert isinstance(img_index, str) and len(img_index) == 2+8+1, img_index
        cam_index = int(img_index[:2])
        time_index = int(img_index[-8:])
        return cam_index, time_index

    def get_image_filename(self, scene_name, img_index, image_type):
        """
        Get the image filename for that scene and image index.
        Image index must be a string %02d_%08d combining the cam and time indices.
        This exists for compatibility with superclass users of this function
        like get_rgbd_mask_pose.
        :param scene_name: str
        :param img_index: str
        :param image_type: ImageType
        :return:
        """

        scene_directory = self.get_full_path_for_scene(scene_name)

        if image_type == ImageType.RGB:
            file_extension = ".jpg"
        elif image_type == ImageType.DEPTH:
            file_extension = "_drake_depth.png"
        elif image_type == ImageType.MASK:
            file_extension = "_drake_label.png"
        else:
            raise ValueError("unsupported image type")

        assert isinstance(img_index, str), (img_index, type(img_index))
        scene_directory = self.get_full_path_for_scene(scene_name)
        if not os.path.isdir(scene_directory):
            raise ValueError("scene_name = %s doesn't exist" %(scene_name))

        return os.path.join(scene_directory, img_index + file_extension)

    def get_image_filename_with_cam_and_time_index(self, scene_name, camera_index, time_index, image_type):
        """
        Get the image filename for that scene, camera, and time index
        :param scene_name: str
        :param camera_index: int
        :param time_index: int
        :param image_type: ImageType
        :return:
        """

        scene_directory = self.get_full_path_for_scene(scene_name)

        if image_type == ImageType.RGB:
            file_extension = ".jpg"
        elif image_type == ImageType.DEPTH:
            file_extension = "_drake_depth.png"
        elif image_type == ImageType.MASK:
            file_extension = "_drake_label.png"
        else:
            raise ValueError("unsupported image type")

        image_string_index = self.make_image_string_index(camera_index, time_index)
        scene_directory = self.get_full_path_for_scene(scene_name)
        if not os.path.isdir(scene_directory):
            raise ValueError("scene_name = %s doesn't exist" %(scene_name))

        return os.path.join(scene_directory, image_string_index + file_extension)
    
    def get_pose_from_scene_name_and_idx(self, scene_name, idx):
        """
        :param scene_name: str
        :param img_idx: string that we can break into camera and time index
        :return: 4 x 4 numpy array
        """
        cam_index, time_index = self.split_image_string_index(idx)
        metadata = self.get_scene_metadata(scene_name)
        cam_quatxyz = list(metadata["data"][time_index]["camera_frames"]["cam_%02d" % cam_index]["pose"])
        transform_matrix = transformations.quaternion_matrix(cam_quatxyz[:4])
        transform_matrix[0:3,3] = np.array(cam_quatxyz[-3:])
        return transform_matrix

    def get_camera_intrinsics(self, scene_name, camera_index):
        """
        Returns the camera matrix for that scene and camera
        :param scene_name: str
        :param camera_index: int
        :return:
        :rtype:
        """
        metadata = self.get_scene_metadata(scene_name)
        camera_config = metadata["cameras"]["cam_%02d" % camera_index]

        # TODO(gizatt) CameraIntrinsics offers a constructor from a yaml
        # filename but not an already-loaded dict.
        fx = camera_config['calibration']['camera_matrix']['data'][0]
        cx = camera_config['calibration']['camera_matrix']['data'][2]
        fy = camera_config['calibration']['camera_matrix']['data'][4]
        cy = camera_config['calibration']['camera_matrix']['data'][5]
        width = camera_config['calibration']['image_width']
        height = camera_config['calibration']['image_height']
        return CameraIntrinsics(cx, cy, fx, fy, width, height)

    def get_random_camera_index_at_time(self, scene_name, time_index, num_distinct_indices=1):
        """
        Returns a random camera index from a given scene and time
        :param scene_name: string
        :param time_index: int
        :return:
        :rtype:
        """
        metadata = self.get_scene_metadata(scene_name)

        scene_view_metadata_list = metadata["data"]
        assert (time_index >=0 and time_index < len(scene_view_metadata_list)), (time_index, len(scene_view_metadata_list))
        num_camera_frames_at_time = len(scene_view_metadata_list[time_index]["camera_frames"].keys())
        possible_inds = range(num_camera_frames_at_time)
        return random.sample(possible_inds, num_distinct_indices)

    def get_random_time_index(self, scene_name):
        """
        Returns a random image index from a given scene
        :param scene_name: string
        :return:
        :rtype:
        """
        metadata = self.get_scene_metadata(scene_name)
        return random.randrange(len(metadata["data"]))

    def get_random_scene_name(self):
        """
        Returns a random multi object scene name
        :return:
        :rtype:
        """
        return random.choice(self._scene_dict[self.mode])

    def get_within_scene_data(self, scene_name):
        """
        The method through which the dataset is accessed for training.

        Each call is is the result of
        a random sampling over:
        - random scene view (i.e. scene + time index)
        - random rgbd frame from that scene view
        - random rgbd frame (different enough pose) from that scene view
        - various randomization in the match generation and non-match generation procedure

        returns a large amount of variables, separated by commas.

        0th return arg: the type of data sampled (this can be used as a flag for different loss functions)
        0th rtype: string

        1st, 2nd return args: image_a_rgb, image_b_rgb
        1st, 2nd rtype: 3-dimensional torch.FloatTensor of shape (image_height, image_width, 3)

        3rd, 4th return args: matches_a, matches_b
        3rd, 4th rtype: 1-dimensional torch.LongTensor of shape (num_matches)

        5th, 6th return args: masked_non_matches_a, masked_non_matches_b
        5th, 6th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        9th, 10th return args: blind_non_matches_a, blind_non_matches_b
        9th, 10th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        11th return arg: metadata useful for plotting, and-or other flags for loss functions
        11th rtype: dict

        Return values 3,4,5,6,7,8,9,10 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        If no datapoints were found for some type of match or non-match then we return
        our "special" empty tensor. Note that due to the way the pytorch data loader
        functions you cannot return an empty tensor like torch.FloatTensor([]). So we
        return SpartanDataset.empty_tensor()

        """

        SCD = SyntheticCorrespondenceDataset

        time_idx = self.get_random_time_index(scene_name)

        image_a_cam_idx, image_b_cam_idx = self.get_random_camera_index_at_time(scene_name, time_idx, num_distinct_indices=2)
        image_a_string_index = self.make_image_string_index(time_idx, image_a_cam_idx)
        image_b_string_index = self.make_image_string_index(time_idx, image_b_cam_idx)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self.get_rgbd_mask_pose(scene_name, image_a_string_index)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self.get_rgbd_mask_pose(scene_name, image_b_string_index)

        # Clamp down the background label (which is very large) to 0
        image_a_mask = np.asarray(image_a_mask).copy()
        image_b_mask = np.asarray(image_b_mask).copy()
        image_a_mask[image_a_mask < 2] = 0
        image_b_mask[image_b_mask < 2] = 0
        image_a_mask[image_a_mask > 10] = 0
        image_b_mask[image_b_mask > 10] = 0
        image_a_mask[image_a_mask != 0] = 1.
        image_b_mask[image_b_mask != 0] = 1.
        
        # And then back to PIL...yuck
        image_a_mask = Image.fromarray(image_a_mask)
        image_b_mask = Image.fromarray(image_b_mask)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        if self.sample_matches_only_off_mask:
            correspondence_mask = np.asarray(image_a_mask)
        else:
            correspondence_mask = None

        # find correspondences
        K_image_a = self.get_camera_intrinsics(scene_name, image_a_cam_idx).get_camera_matrix()
        K_image_b = self.get_camera_intrinsics(scene_name, image_b_cam_idx).get_camera_matrix()
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose,
                                                                            image_b_depth_numpy, image_b_pose,
                                                                            img_a_mask=correspondence_mask,
                                                                            num_attempts=self.num_matching_attempts,
                                                                            K_a=K_image_a,
                                                                            K_b=K_image_b)
        if uv_a is None:
            logging.info("no matches found, returning")
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)


        # data augmentation
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)

        if not self.debug:
            [image_a_rgb, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_mask], uv_a)
            [image_b_rgb, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_mask], uv_b)
        else:  # also mutate depth just for plotting
            [image_a_rgb, image_a_depth, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation(
                [image_a_rgb, image_a_depth, image_a_mask], uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_depth, image_b_mask], uv_b)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)


        # find non_correspondences
        image_b_mask_torch = torch.from_numpy(np.asarray(image_b_mask)).type(torch.FloatTensor)
        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]

        uv_b_masked_non_matches = \
            correspondence_finder.create_non_correspondences(uv_b,
                                                             image_b_shape,
                                                             num_non_matches_per_match=self.num_masked_non_matches_per_match,
                                                                            img_b_mask=image_b_mask_torch)


        if self._use_image_b_mask_inv:
            image_b_mask_inv = 1 - image_b_mask_torch
        else:
            image_b_mask_inv = None

        uv_b_background_non_matches = correspondence_finder.create_non_correspondences(uv_b,
                                                                            image_b_shape,
                                                                            num_non_matches_per_match=self.num_background_non_matches_per_match,
                                                                            img_b_mask=image_b_mask_inv)



        # convert PIL.Image to torch.FloatTensor
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        matches_a = SCD.flatten_uv_tensor(uv_a, image_width)
        matches_b = SCD.flatten_uv_tensor(uv_b, image_width)

        # Masked non-matches
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(uv_a, uv_b_masked_non_matches, self.num_masked_non_matches_per_match)

        masked_non_matches_a = SCD.flatten_uv_tensor(uv_a_masked_long, image_width).squeeze(1)
        masked_non_matches_b = SCD.flatten_uv_tensor(uv_b_masked_non_matches_long, image_width).squeeze(1)


        # Non-masked non-matches
        uv_a_background_long, uv_b_background_non_matches_long = self.create_non_matches(uv_a, uv_b_background_non_matches,
                                                                            self.num_background_non_matches_per_match)

        background_non_matches_a = SCD.flatten_uv_tensor(uv_a_background_long, image_width).squeeze(1)
        background_non_matches_b = SCD.flatten_uv_tensor(uv_b_background_non_matches_long, image_width).squeeze(1)


        # make blind non matches
        matches_a_mask = SCD.mask_image_from_uv_flat_tensor(matches_a, image_width, image_height)
        image_a_mask_torch = torch.from_numpy(np.asarray(image_a_mask)).long()
        mask_a_flat = image_a_mask_torch.view(-1,1).squeeze(1)
        blind_non_matches_a = (mask_a_flat - matches_a_mask).nonzero()

        no_blind_matches_found = False
        if len(blind_non_matches_a) == 0:
            no_blind_matches_found = True
        else:

            blind_non_matches_a = blind_non_matches_a.squeeze(1)
            num_blind_samples = blind_non_matches_a.size()[0]

            if num_blind_samples > 0:
                # blind_uv_b is a tuple of torch.LongTensor
                # make sure we check that blind_uv_b is not None and that it is non-empty


                blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(image_b_mask_torch, num_blind_samples)

                if blind_uv_b[0] is None:
                    no_blind_matches_found = True
                elif len(blind_uv_b[0]) == 0:
                    no_blind_matches_found = True
                else:
                    blind_non_matches_b = utils.uv_to_flattened_pixel_locations(blind_uv_b, image_width)

                    if len(blind_non_matches_b) == 0:
                        no_blind_matches_found = True
            else:
                no_blind_matches_found = True

        if no_blind_matches_found:
            blind_non_matches_a = blind_non_matches_b = SCD.empty_tensor()


        if self.debug:
            # downsample so can plot
            num_matches_to_plot = 10
            plot_uv_a, plot_uv_b = SCD.subsample_tuple_pair(uv_a, uv_b, num_samples=num_matches_to_plot)

            plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long = SCD.subsample_tuple_pair(uv_a_masked_long, uv_b_masked_non_matches_long, num_samples=num_matches_to_plot*3)

            plot_uv_a_background_long, plot_uv_b_background_non_matches_long = SCD.subsample_tuple_pair(uv_a_background_long, uv_b_background_non_matches_long, num_samples=num_matches_to_plot*3)

            blind_uv_a = utils.flattened_pixel_locations_to_u_v(blind_non_matches_a, image_width)
            plot_blind_uv_a, plot_blind_uv_b = SCD.subsample_tuple_pair(blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot*10)


        if self.debug:
            # only want to bring in plotting code if in debug mode
            import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter

            # Show correspondences
            if uv_a is not None:
                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               plot_uv_a, plot_uv_b, show=False)

                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='r')

                fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               plot_uv_a, plot_uv_b, show=False)

                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_uv_a_background_long, plot_uv_b_background_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='b')


                correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_blind_uv_a, plot_blind_uv_b,
                                                                   circ_color='k', show=True)

                # Mask-plotting city
                import matplotlib.pyplot as plt
                plt.imshow(np.asarray(image_a_mask))
                plt.title("Mask of img a object pixels")
                plt.show()

                plt.imshow(np.asarray(image_a_mask) - 1)
                plt.title("Mask of img a background")
                plt.show()

                temp = matches_a_mask.view(image_height, -1)
                plt.imshow(temp)
                plt.title("Mask of img a object pixels for which there was a match")
                plt.show()

                temp2 = (mask_a_flat - matches_a_mask).view(image_height, -1)
                plt.imshow(temp2)
                plt.title("Mask of img a object pixels for which there was NO match")
                plt.show()



        return image_a_rgb, image_b_rgb, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, blind_non_matches_a, blind_non_matches_b

    def create_non_matches(self, uv_a, uv_b_non_matches, multiplier):
        """
        Simple wrapper for repeated code
        :param uv_a:
        :type uv_a:
        :param uv_b_non_matches:
        :type uv_b_non_matches:
        :param multiplier:
        :type multiplier:
        :return:
        :rtype:
        """
        uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                     torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

        uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

        return uv_a_long, uv_b_non_matches_long

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

    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """

        return self._rgb_image_to_tensor(img.copy())

    @property
    def config(self):
        return self._config
    
    @staticmethod
    def flatten_uv_tensor(uv_tensor, image_width):
        """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
        """
        return uv_tensor[1].long() * image_width + uv_tensor[0].long()

    @staticmethod
    def mask_image_from_uv_flat_tensor(uv_flat_tensor, image_width, image_height):
        """
        Returns a torch.LongTensor with shape [image_width*image_height]. It has a 1 exactly
        at the indices specified by uv_flat_tensor
        :param uv_flat_tensor:
        :type uv_flat_tensor:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :return:
        :rtype:
        """
        image_flat = torch.zeros(image_width*image_height).long()
        image_flat[uv_flat_tensor] = 1
        return image_flat


    @staticmethod
    def subsample_tuple(uv, num_samples):
        """
        Subsamples a tuple of (torch.Tensor, torch.Tensor)
        """
        indexes_to_keep = (torch.rand(num_samples) * len(uv[0])).floor().type(torch.LongTensor)
        return (torch.index_select(uv[0], 0, indexes_to_keep), torch.index_select(uv[1], 0, indexes_to_keep))

    @staticmethod
    def subsample_tuple_pair(uv_a, uv_b, num_samples):
        """
        Subsamples a pair of tuples, i.e. (torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)
        """
        assert len(uv_a[0]) == len(uv_b[0])
        indexes_to_keep = (torch.rand(num_samples) * len(uv_a[0])).floor().type(torch.LongTensor)
        uv_a_downsampled = (torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep))
        uv_b_downsampled = (torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep))
        return uv_a_downsampled, uv_b_downsampled
