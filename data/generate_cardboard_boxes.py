import lxml.etree as et
import skimage.draw
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import trimesh
from trimesh.constants import log
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
import PIL.Image
import scipy
import scipy.interpolate
import cv2
import time
import sys

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
    uv_scale_factor = np.max(uvs)
    uvs /= uv_scale_factor # Make the max dimension 1

    uvs_by_face = {
        'nx': uvs[0:4, :],
        'px': uvs[4:8, :],
        'ny': uvs[8:12, :],
        'py': uvs[12:16, :],
        'nz': uvs[16:20, :],
        'pz': uvs[20:24, :]
    }

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        visual=trimesh.visual.texture.TextureVisuals(uv=uvs),
        process=False, # Processing / validation removes duplicate verts, ruining our UV mapping
        validate=False)
    return mesh, uvs_by_face, uv_scale_factor


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
    mask = cv2.warpPerspective(np.ones(decal_image.shape[:2]), M, (base_image.shape[0], base_image.shape[1]))

    # Swap back to numpy axis ordering
    src_rgba = np.ascontiguousarray(np.swapaxes(dst, 0, 1).astype(np.float32)/255.)
    mask = np.swapaxes(mask, 0, 1)

    # Ref https://stackoverflow.com/questions/25182421/overlay-two-numpy-arrays-treating-fourth-plane-as-alpha-level
    src_rgb = src_rgba[..., :3]
    src_a = src_rgba[..., 3] * mask
    dst_rgba = np.ascontiguousarray(base_image.astype(np.float32)/255.)
    dst_rgb = dst_rgba[..., :3]
    dst_a = dst_rgba[..., 3]
    dst_a_rel = dst_a*(1.0-src_a)
    out_a = src_a + dst_a_rel
    out_rgb = (src_rgb*src_a[..., None]
           + dst_rgb*dst_a_rel[..., None]) / out_a[..., None]
    base_image[:, :, :3] = out_rgb * 255
    base_image[:, :, 3] = out_a * 255

    
def tile_box_with_texture(
        base_image, decal_image, corners,
        scale=(1.0, 1.0),
        number_of_tiles=None):
    '''
        base_image: a X_b by Y_b by C pixel image to be modified.
        decal_image: an X_d by Y_d by C pixel image to be drawn into the specified box.
        Corners: A list of 4 UV coordinates (range 0-1)
        indicating the lower left, upper left, upper right,
        and lower right corners of where the image should go.
    '''
    start_time = time.time()

    # Rescale the declar first
    scaling = np.ones(3)
    scaling[:2] = scale
    decal_image_scaled = scipy.ndimage.zoom(decal_image, scaling, mode='wrap', order=0)

    if number_of_tiles is None:
        number_of_tiles = (np.max(corners, axis=0) - np.min(corners, axis=0)) * \
            base_image.shape[:2] / decal_image_scaled.shape[:2]
    # Tile the decal image a sufficient number of times
    reps = np.hstack([np.ceil(number_of_tiles).astype(int), 1])
    decal_full = np.tile(decal_image_scaled, reps=reps)
    # And clip it down to the exact number of pixels in each direction it should be
    true_size = np.ceil(decal_full.shape[:2] * (number_of_tiles / np.ceil(number_of_tiles))).astype(int)
    decal_full = decal_full[:true_size[0], :true_size[1], :]
    fill_box_with_texture(base_image, decal_full, corners)
    print("\tDid tiling in %f seconds" % (time.time() - start_time))

if __name__ == "__main__":

    # attach to logger so trimesh messages will be printed to console
    # trimesh.util.attach_to_log()

    sx, sy, sz = np.random.uniform(low=0.2, high=0.6, size=(3,))
    print("Box shape: ", sx, sy, sz)
    mesh, box_uvs_by_face, box_uv_scale_factor = generated_scaled_box_with_uvs(sx, sy, sz)
    print("Box uv scale factor: ", box_uv_scale_factor)

    # Generate a base texture for the box using a cardboard texture
    baseColorTexture = np.zeros((2048, 2048, 4), dtype=np.uint8)

    # Fill the base color map with the cardboard texture
    cardboard_texture = np.array(PIL.Image.open("/home/gizatt/data/cardboard_box_texturing/textures/cardboard_tileable_1.png"))[:, :, :4]
    #cardboard_texture = np.array(PIL.Image.open("/home/gizatt/Downloads/grid.jpg"))
    tile_box_with_texture(baseColorTexture, cardboard_texture, np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                          scale=[0.1, 0.1])
    #sys.exit()
    # Apply a strip of tape with some noise along the long tile direction
    for k in range(1):
        possible_tape_textures = [
            "/home/gizatt/data/cardboard_box_texturing/textures/amazon_prime_tape_tileable.png",
            "/home/gizatt/data/cardboard_box_texturing/textures/tape_sample.png",
        ]
        tape_texture = np.array(PIL.Image.open(random.choice(possible_tape_textures)))
        if tape_texture.shape[2] != 4:
            assert(tape_texture.shape[2] == 3)
            tape_texture = np.stack([tape_texture, np.ones(tape_texture.shape[:2])], dim=3)
        tape_width_m = 0.07
        tape_uv_width = tape_width_m / box_uv_scale_factor
        tape_uv_length = 1.6 # Long enough to always wrap all the way
        pz_mean = np.mean(box_uvs_by_face['pz'], axis=0)
        tape_corners_pre_rotation = np.array([[-tape_uv_length/2., -tape_uv_width/2.],
                                              [tape_uv_length/2., -tape_uv_width/2.],
                                              [tape_uv_length/2., tape_uv_width/2.],
                                              [-tape_uv_length/2., tape_uv_width/2.]])
        rotation = np.random.uniform(-0.1, 0.1)
        rotmat = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        # Why is this flip needed?
        offset = np.array([pz_mean[1], pz_mean[0]]) + np.random.randn(2)*0.01
        corners = np.dot(rotmat, tape_corners_pre_rotation.T).T + offset
        # Scale so exactly 1 repetition width-wise of the tape will happen
        scaling = np.ones(2) * (baseColorTexture.shape[1] * tape_uv_width) / tape_texture.shape[1]
        tile_box_with_texture(baseColorTexture, tape_texture, corners, scale=scaling,
                              number_of_tiles=[tape_uv_length*tape_texture.shape[1]/(tape_uv_width*tape_texture.shape[0]), 1.])

    for k in range(5):
        # Apply other labels totally randomly
        possible_labels = [
            '/home/gizatt/data/cardboard_box_texturing/textures/bar_code_decal_1.png',
            #'/home/gizatt/data/cardboard_box_texturing/textures/bar_code_sticker_decal_1.png',
            '/home/gizatt/data/cardboard_box_texturing/textures/recycleable_decal.png',
            '/home/gizatt/data/cardboard_box_texturing/textures/sticker_placement_decal.png',
        ]
        sticker_texture = np.array(PIL.Image.open(random.choice(possible_labels)))
        u_scale = np.random.uniform(0.02, 0.15)
        v_scale = u_scale * sticker_texture.shape[1] / sticker_texture.shape[0]
        corners_pre_rotation = np.array([[-u_scale/2., -v_scale/2.],
                                              [u_scale/2., -v_scale/2.],
                                              [u_scale/2., v_scale/2.],
                                              [-u_scale/2., v_scale/2.]])
        rotation = np.random.randint(4)*np.pi/2. + np.random.randn()*0.05
        rotmat = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        # Stick it randomly on a random face
        uv_bounds = random.choice(list(box_uvs_by_face.values()))
        offset = random.uniform(np.min(uv_bounds, axis=0), np.max(uv_bounds, axis=0))
        offset = np.array([offset[1], offset[0]]) # why???

        corners = np.dot(rotmat, corners_pre_rotation.T).T + offset
        scaling = np.ones(2) * (baseColorTexture.shape[1] * v_scale) / sticker_texture.shape[1]
        print("Sticker at corners: ", corners)
        tile_box_with_texture(baseColorTexture, sticker_texture, corners, scale=scaling,
                              number_of_tiles=[1, 1])

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