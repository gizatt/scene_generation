import os
import sys

import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss

# This needs to be after dense correspondence does its imports
# so it finds it preferred version of torchvision
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

import numpy as np
import matplotlib.pyplot as plt
from synthetic_correspondence_dataset import SyntheticCorrespondenceDataset


def zero_loss():
    return Variable(torch.FloatTensor([0]).cuda())

def is_zero_loss(loss):
    return loss.item() < 1e-20

def get_within_scene_loss(pixelwise_contrastive_loss, image_a_pred, image_b_pred,
                                        matches_a,    matches_b,
                                        masked_non_matches_a, masked_non_matches_b,
                                        background_non_matches_a, background_non_matches_b,
                                        blind_non_matches_a, blind_non_matches_b):
    """
    Simple wrapper for pixelwise_contrastive_loss functions.  Args and return args documented above in get_loss()
    """
    pcl = pixelwise_contrastive_loss

    match_loss, masked_non_match_loss, num_masked_hard_negatives =\
        pixelwise_contrastive_loss.get_loss_matched_and_non_matched_with_l2(image_a_pred,         image_b_pred,
                                                                          matches_a,            matches_b,
                                                                          masked_non_matches_a, masked_non_matches_b,
                                                                          M_descriptor=pcl._config["M_masked"])

    if pcl._config["use_l2_pixel_loss_on_background_non_matches"]:
        background_non_match_loss, num_background_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_with_l2_pixel_norm(image_a_pred, image_b_pred, matches_b, 
                background_non_matches_a, background_non_matches_b, M_descriptor=pcl._config["M_background"])    
        
    else:
        background_non_match_loss, num_background_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    background_non_matches_a, background_non_matches_b,
                                                                    M_descriptor=pcl._config["M_background"])
        
        

    blind_non_match_loss = zero_loss()
    num_blind_hard_negatives = 1
    if not (SyntheticCorrespondenceDataset.is_empty(blind_non_matches_a.data)):
        blind_non_match_loss, num_blind_hard_negatives =\
            pixelwise_contrastive_loss.non_match_loss_descriptor_only(image_a_pred, image_b_pred,
                                                                    blind_non_matches_a, blind_non_matches_b,
                                                                    M_descriptor=pcl._config["M_masked"])
        


    total_num_hard_negatives = num_masked_hard_negatives + num_background_hard_negatives
    total_num_hard_negatives = max(total_num_hard_negatives, 1)

    if pcl._config["scale_by_hard_negatives"]:
        scale_factor = total_num_hard_negatives

        masked_non_match_loss_scaled = masked_non_match_loss*1.0/max(num_masked_hard_negatives, 1)

        background_non_match_loss_scaled = background_non_match_loss*1.0/max(num_background_hard_negatives, 1)

        blind_non_match_loss_scaled = blind_non_match_loss*1.0/max(num_blind_hard_negatives, 1)
    else:
        # we are not currently using blind non-matches
        num_masked_non_matches = max(len(masked_non_matches_a),1)
        num_background_non_matches = max(len(background_non_matches_a),1)
        num_blind_non_matches = max(len(blind_non_matches_a),1)
        scale_factor = num_masked_non_matches + num_background_non_matches


        masked_non_match_loss_scaled = masked_non_match_loss*1.0/num_masked_non_matches

        background_non_match_loss_scaled = background_non_match_loss*1.0/num_background_non_matches

        blind_non_match_loss_scaled = blind_non_match_loss*1.0/num_blind_non_matches



    non_match_loss = 1.0/scale_factor * (masked_non_match_loss + background_non_match_loss)

    loss = pcl._config["match_loss_weight"] * match_loss + \
    pcl._config["non_match_loss_weight"] * non_match_loss

    return loss, match_loss, masked_non_match_loss_scaled, background_non_match_loss_scaled, blind_non_match_loss_scaled

if __name__ == "__main__":
    print("Worked")

    data_config_filename = "dataset_config.yaml"
    data_config = utils.getDictFromYamlFilename(data_config_filename)
    dataset = SyntheticCorrespondenceDataset(data_config, debug=False)
    
    train_config_file = os.path.join(
        utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
        'training', 'training.yaml')
    train_config = utils.getDictFromYamlFilename(train_config_file)
    batch_size = train_config['training']['batch_size']

    dcn = DenseCorrespondenceNetwork.from_config(
        train_config['dense_correspondence_network'],
        load_stored_params=False,
        model_param_file=None)
    print("Build the DCN: ", dcn)

    dataset.set_parameters_from_training_config(train_config)

    pixelwise_contrastive_loss = PixelwiseContrastiveLoss(
        image_shape=dcn.image_shape,
        config=train_config['loss_function'])

    # Load data, which has fields
    (img_a, img_b, matches_a, matches_b,
     masked_non_matches_a, masked_non_matches_b,
     background_non_matches_a, background_non_matches_b,
     blind_non_matches_a, blind_non_matches_b) = dataset[0]

    # Load two test images for kicks

    img_a_pil = Image.open("/home/gizatt/data/generated_cardboard_envs/scene_group_5_types_100/scene_000/00_00000000.jpg")
    img_b_pil = Image.open("/home/gizatt/data/generated_cardboard_envs/scene_group_5_types_100/scene_000/01_00000000.jpg")
    img_a = ToTensor()(img_a_pil).unsqueeze(0)
    img_b = ToTensor()(img_b_pil).unsqueeze(0)
    img_a = Variable(img_a.cuda(), requires_grad=False)
    img_b = Variable(img_b.cuda(), requires_grad=False)

    # Evaluate
    image_a_pred = dcn.forward(img_a)
    image_a_pred_post = dcn.process_network_output(image_a_pred, batch_size)

    image_b_pred = dcn.forward(img_b)
    image_b_pred_post = dcn.process_network_output(image_b_pred, batch_size)

    print(image_a_pred.shape)
    image_a_pred_pil = ToPILImage()(image_a_pred.cpu().squeeze(0))
    image_b_pred_pil = ToPILImage()(image_b_pred.cpu().squeeze(0))

    matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
    matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
    masked_non_matches_a = Variable(masked_non_matches_a.cuda().squeeze(0), requires_grad=False)
    masked_non_matches_b = Variable(masked_non_matches_b.cuda().squeeze(0), requires_grad=False)

    background_non_matches_a = Variable(background_non_matches_a.cuda().squeeze(0), requires_grad=False)
    background_non_matches_b = Variable(background_non_matches_b.cuda().squeeze(0), requires_grad=False)

    blind_non_matches_a = Variable(blind_non_matches_a.cuda().squeeze(0), requires_grad=False)
    blind_non_matches_b = Variable(blind_non_matches_b.cuda().squeeze(0), requires_grad=False)

    # get loss
    loss, match_loss, masked_non_match_loss, \
    background_non_match_loss, blind_non_match_loss = get_within_scene_loss(
        pixelwise_contrastive_loss,
        image_a_pred_post, image_b_pred_post,
        matches_a,     matches_b,
        masked_non_matches_a, masked_non_matches_b,
        background_non_matches_a, background_non_matches_b,
        blind_non_matches_a, blind_non_matches_b)
    
    print("Loss: ", loss)
    print("Loss elements: ", match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Before A")
    plt.imshow(img_a_pil)
    plt.title("Before B")
    plt.subplot(2, 2, 2)
    plt.imshow(img_b_pil)
    plt.subplot(2, 2, 3)
    plt.title("After A")
    plt.imshow(image_a_pred_pil)
    plt.title("After B")
    plt.subplot(2, 2, 4)
    plt.imshow(image_b_pred_pil)
    plt.show()
    
    