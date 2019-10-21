import lxml.etree as et
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from trimesh.constants import log
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
import PIL.Image
import scipy
import cv2

import os

import numpy as np

def generate_unit_box_face_uvs():
    # Using the default UV unwrapping from blender for the unit cube, each face gets
    # projected like:
    top_uv_corners = np.array(
        [[0.25, 0.25 + 0.125],
         [0.25, 0.5 + 0.125],
         [0.5, 0.5 + 0.125],
         [0.5, 0.25 + 0.125]])

    bottom_uv_corners = np.array(
        [[0.75, 0.25 + 0.125],
         [0.75, 0.5 + 0.125],
         [1.0, 0.5 + 0.125],
         [1.0, 0.25 + 0.125]])

    front_uv_corners = np.array(
        [[0.0, 0.25 + 0.125],
         [0.0, 0.5 + 0.125],
         [0.25, 0.5 + 0.125],
         [0.25, 0.25 + 0.125]])

    back_uv_corners = np.array(
        [[0.5, 0.25 + 0.125],
         [0.5, 0.5 + 0.125],
         [0.75, 0.5 + 0.125],
         [0.75, 0.25 + 0.125]])

    left_uv_corners = np.array(
        [[0.25, 0. + 0.125],
         [0.25, 0.25 + 0.125],
         [0.5, 0.25 + 0.125],
         [0.5, 0. + 0.125]])

    right_uv_corners = np.array(
        [[0.25, 0.5 + 0.125],
         [0.25, 0.75 + 0.125],
         [0.5, 0.75 + 0.125],
         [0.5, 0.5 + 0.125]])

    face_uvs = {
        "top": top_uv_corners,
        "bottom": bottom_uv_corners,
        "front": front_uv_corners,
        "back": back_uv_corners,
        "left": left_uv_corners,
        "right": right_uv_corners
    }

    return face_uvs

def fill_box_with_texture(base_image, decal_image, corners):
    '''
        base_image: a X_b by Y_b by C pixel image to be modified.
        decal_image: an X_d by Y_d by C pixel image to be drawn into the specified box.
        Corners: A list of 4 UV coordinates (range 0-1)
        indicating the lower left, upper left, upper right,
        and lower right corners of where the image should go.
    '''

    # These are OpenCV points, which are column-row ordering.
    corners_base = corners.copy().astype(np.float32)
    corners_base[:, 0] *= base_image.shape[1]
    corners_base[:, 1] *= base_image.shape[0]
    corners_decal = np.array([[0., 0.],
                              [0., 1.],
                              [1., 1.],
                              [1., 0.]]).astype(np.float32)
    corners_decal[:, 0] *= decal_image.shape[1]
    corners_decal[:, 1] *= decal_image.shape[0]

    M = cv2.getPerspectiveTransform(
        corners_decal,
        corners_base)
    dst = cv2.warpPerspective(decal_image, M, (base_image.shape[0], base_image.shape[1]))
    mask = cv2.warpPerspective(np.ones(decal_image.shape), M, (base_image.shape[0], base_image.shape[1]))
    # Swap back to numpy axis ordering
    dst = np.swapaxes(dst, 0, 1)
    mask = np.swapaxes(mask, 0, 1)
    base_image[:] = np.where(mask, dst, base_image)

def tile_box_with_texture(base_image, decal_image, corners, scale=(1.0, 1.0), rotation=0.0):
    '''
        base_image: a X_b by Y_b by C pixel image to be modified.
        decal_image: an X_d by Y_d by C pixel image to be drawn into the specified box.
        Corners: A list of 4 UV coordinates (range 0-1)
        indicating the lower left, upper left, upper right,
        and lower right corners of where the image should go.
    '''

    number_of_tiles = (np.max(corners, axis=0) - np.min(corners, axis=0)) * \
        base_image.shape[:2] / decal_image.shape[:2] / np.array(scale)
    # Tile the decal image a sufficient number of times
    decal_full = np.tile(decal_image, reps=np.ceil(number_of_tiles).astype(int))
    # And clip it down to the exact number of pixels in each direction it should be
    true_size = np.ceil(decal_full.shape[:2] * (number_of_tiles / np.ceil(number_of_tiles))).astype(int)
    decal_full = decal_full[:true_size[0], :true_size[1], :]
    # Stretch to fix the aspect ratio
    #aspect_ratio = number_of_tiles[0] / number_of_tiles[1]
    #decal_full = scipy.ndimage.zoom(decal_full, zoom=[1., aspect_ratio], mode='wrap')
    decal_rotated = scipy.ndimage.rotate(
        decal_full, rotation, reshape=False, mode='wrap')

    fill_box_with_texture(base_image, decal_rotated, corners)

if __name__ == "__main__":

    # attach to logger so trimesh messages will be printed to console
   # trimesh.util.attach_to_log()

    mesh = trimesh.load_mesh('/home/gizatt/data/cardboard_box_texturing/unit_box_beveled.obj')
    
    # Rescale the box
    extents = np.array([1.0, 1.0, 0.5])
    for k in range(len(extents)):
        mesh.vertices[:, k] *= extents[k]

    face_uvs = generate_unit_box_face_uvs()

    color = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Fill each side with a unique color
    cmap = plt.cm.get_cmap('Spectral')
    for i, corner_uvs in enumerate(face_uvs.values()):
        min_x = int(min(corner_uvs[:, 0])*color.shape[0])
        max_x = int(max(corner_uvs[:, 0])*color.shape[0])
        min_y = int(min(corner_uvs[:, 1])*color.shape[1])
        max_y = int(max(corner_uvs[:, 1])*color.shape[1])
        color[min_x:max_x, min_y:max_y, :] = np.array(cmap(float(i)/len(face_uvs.keys()))[:3])*255

    baseColorTexture = np.array(PIL.Image.open("/home/gizatt/data/cardboard_box_texturing/textures/cardboard_tileable_1.png"))[:, :, :3]
    #baseColorTexture = np.array(PIL.Image.open("/home/gizatt/Dropbox/Misc/desktops_work/check-108 (1).png"))[:, :, :3]
    tile_box_with_texture(color, baseColorTexture, face_uvs['top'],
                          scale=np.array([1./extents[1], 1/extents[2]])/10.)

    baseColorTexture = PIL.Image.fromarray(color)
    #baseColorTexture = PIL.Image.open("cardboard-effect-and-colour-paper.jpg")

    mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=baseColorTexture)
    mesh.show()
