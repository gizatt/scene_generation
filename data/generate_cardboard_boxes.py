import lxml.etree as et
import skimage.draw
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import trimesh
from trimesh.constants import log
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
import PIL.Image
import scipy
import scipy.interpolate
import cv2

import os

import numpy as np

def generated_scaled_box_with_uvs(sx, sy, sz):
    # Hand-code generation of the box, so I can get the UV unwrapping
    # correct and manually scaled.
    # Intended unwrapping is:         
    #                              v
    #         +---------------------------------------->
    #
    #              sz         sx       sz        sx
    #    +    +----------------------------------------+
    #    |    |          |         |                   |
    #    |    |          |         |                   |
    #    |  s |          |    +y   |                   |
    #    |  z |          |         |                   |
    #    |    |          |         |                   |
    # u  |    +----------------------------------------+
    #    |    |          |         |         |         |
    #    |  s |    -x    |    +z   |    +x   |   -z    |
    #    |  y |          |         |         |         |
    #    |    |          |         |         |         |
    #    |    +----------------------------------------+
    #    |    |          |         |                   |
    #    |  s |          |    -y   |                   |
    #    |  z |          |         |                   |
    #    |    |          |         |                   |
    #    v    +----------------------------------------+
    #
    # (u vertical, v horizontal, sx/sy/sz labels indicate
    # the width of each chunk)

    verts = []
    normals = []
    faces = []
    uvs = []

    ### -X
    u_lb = sz
    v_lb = 0.
    u_s = sy
    v_s = sz
    verts += [[-sx, sy, -sz],
              [-sx, -sy, -sz],
              [-sx, -sy, sz],
              [-sx, sy, sz]]
    normals += [[-1, 0., 0.] * 4]
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]

    ### +X
    u_lb = sz
    v_lb = sz + sx
    u_s = sy
    v_s = sz
    verts += [[sx, sy, sz],
              [sx, -sy, sz],
              [sx, -sy, -sz],
              [sx, sy, -sz]]
    normals += [[1, 0., 0.] * 4]
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]

    ### -Y
    u_lb = sz + sy
    v_lb = sz
    u_s = sz
    v_s = sx
    verts += [[-sx, -sy, sz],
              [-sx, -sy, -sz],
              [sx, -sy, -sz],
              [sx, -sy, sz]]
    normals += [[0., -1., 0.] * 4]
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]

    ### +Y
    u_lb = 0.
    v_lb = sz
    u_s = sz
    v_s = sx
    verts += [[-sx, sy, -sz],
              [-sx, sy, sz],
              [sx, sy, sz],
              [sx, sy, -sz]]
    normals += [[0., 1., 0.] * 4]
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]

    ### -Z
    u_lb = sz
    v_lb = sz + sx + sz
    u_s = sy
    v_s = sx
    verts += [[sx, sy, -sz],
              [sx, -sy, -sz],
              [-sx, -sy, -sz],
              [-sx, sy, -sz]]
    normals += [[0., 0., -1.] * 4]
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]

    ### +Z
    u_lb = sz
    v_lb = sz
    u_s = sy
    v_s = sx
    verts += [[-sx, sy, sz],
              [-sx, -sy, sz],
              [sx, -sy, sz],
              [sx, sy, sz]]
    normals += [[0., 0., 1.] * 4]
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]


    verts = np.array(verts)
    faces = np.array(faces)
    normals = np.array(normals)
    uvs = np.array(uvs)
    uvs /= np.max(uvs) # Make the max dimension 1
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        visual=trimesh.visual.texture.TextureVisuals(uv=uvs),
        process=False, # Processing / validation removes duplicate verts, ruining our UV mapping
        validate=False)
    return mesh


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

    print("Decal image shape: ", decal_image.shape)

    # Rescale the declar first
    scaling = np.ones(3)
    scaling[:2] = scale
    decal_image_scaled = scipy.ndimage.zoom(decal_image, scaling, mode='wrap', order=0)

    print("Zoomed shape: ", decal_image_scaled.shape)

    number_of_tiles = (np.max(corners, axis=0) - np.min(corners, axis=0)) * \
        base_image.shape[:2] / decal_image_scaled.shape[:2]
    # Tile the decal image a sufficient number of times
    reps = np.hstack([np.ceil(number_of_tiles).astype(int), 1])
    print("Reps:", reps)
    decal_full = np.tile(decal_image_scaled, reps=reps)
    # And clip it down to the exact number of pixels in each direction it should be
    true_size = np.ceil(decal_full.shape[:2] * (number_of_tiles / np.ceil(number_of_tiles))).astype(int)
    decal_full = decal_full[:true_size[0], :true_size[1], :]
    # Stretch to fix the aspect ratio
    #aspect_ratio = number_of_tiles[0] / number_of_tiles[1]
    #decal_full = scipy.ndimage.zoom(decal_full, zoom=[1., aspect_ratio], mode='wrap')
    decal_rotated = scipy.ndimage.rotate(
        decal_full, rotation, reshape=False, mode='wrap', order=0)

    fill_box_with_texture(base_image, decal_rotated, corners)

if __name__ == "__main__":

    # attach to logger so trimesh messages will be printed to console
    # trimesh.util.attach_to_log()

    sx = 1.
    sy = 3.0
    sz = 0.5
    mesh = generated_scaled_box_with_uvs(sx, sy, sz)


    print("Here1")

    # Generate a base texture for the box using a cardboard texture
    baseColorTexture = np.zeros((2048, 2048, 3), dtype=np.uint8)

    # Fill the base color map with the cardboard texture
    cardboard_texture = np.array(PIL.Image.open("/home/gizatt/data/cardboard_box_texturing/textures/cardboard_tileable_1.png"))[:, :, :3]
    #cardboard_texture = np.array(PIL.Image.open("/home/gizatt/Downloads/grid.jpg"))
    tile_box_with_texture(baseColorTexture, cardboard_texture, np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                          scale=[0.1, 0.1], rotation=0.0)
    
    # Draw the completed texture, with the UV map overlayed
    plt.figure()
    plt.imshow(baseColorTexture)
    for k, face in enumerate(mesh.faces):
        uvs = np.vstack([mesh.visual.uv[v] for v in face])*baseColorTexture.shape[:2]
        patch = patches.Polygon(
            uvs, fill=False, linewidth=1.0, linestyle="--",
            edgecolor=plt.cm.jet(float(k) / len(mesh.faces)))
        plt.gca().add_patch(patch)
    plt.show()



    baseColorTexture = PIL.Image.fromarray(baseColorTexture)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=baseColorTexture)
    mesh.show()