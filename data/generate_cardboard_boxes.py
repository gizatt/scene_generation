from collections import namedtuple
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
import PIL.ImageDraw
import PIL.ImageFont
import scipy
import scipy.interpolate
import string
import cv2
import time
import sys
import os
import yaml

from trimesh_export_obj_with_uv import export_obj


def generate_scaled_box_with_uvs(sx, sy, sz):
    # sx, sy, and sz are *half* widths of faces
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
    normals += [[-1, 0., 0.]] * 4
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
    normals += [[1, 0., 0.]] * 4
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
    normals += [[0., -1., 0.]] * 4
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
    normals += [[0., 1., 0.]] * 4
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
    normals += [[0., 0., -1.]] * 4
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
    normals += [[0., 0., 1.]] * 4
    uvs += [[u_lb, v_lb],
            [u_lb + u_s, v_lb],
            [u_lb + u_s, v_lb + v_s],
            [u_lb, v_lb + v_s]]
    faces += [range(len(verts)-4, len(verts))]

    print("Normals: ", normals)
    verts = np.array(verts)
    faces = np.array(faces)
    normals = np.array(normals)
    uvs = np.array(uvs)
    uv_scale_factor = np.max(uvs)
    uvs /= uv_scale_factor # Make the max dimension 1

    verts_by_face = {
        'nx': verts[0:4, :],
        'px': verts[4:8, :],
        'ny': verts[8:12, :],
        'py': verts[12:16, :],
        'nz': verts[16:20, :],
        'pz': verts[20:24, :]
    }
    normals_by_face = {
        'nx': normals[0, :],
        'px': normals[4, :],
        'ny': normals[8, :],
        'py': normals[12, :],
        'nz': normals[16, :],
        'pz': normals[20, :]
    }
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
    return mesh, uvs_by_face, uv_scale_factor, verts_by_face, normals_by_face

def convert_mesh_uv_to_xyz(mesh, uvs):
    # For each uv, decide the face it lives on
    # (todo(gizatt) for large meshes, rendering a discretized uv-to-face mapping
    # might be more efficient than this exhaustive search)
    xyzs = np.zeros((uvs.shape[0], 3))
    normals = np.zeros((uvs.shape[0], 3))
    faces = np.zeros(uvs.shape[0], dtype=np.int) - 1
    for k in range(uvs.shape[0]):
        uv = uvs[k, :]
        for face_i, face in enumerate(mesh.faces):
            face_uvs = np.vstack([mesh.visual.uv[i, :] for i in face])
            diffs = face_uvs - np.roll(face_uvs, shift=1, axis=0)
            norms = np.vstack([-diffs[:, 1], diffs[:, 0]]).T
            inclusions = np.sum((uv - face_uvs) * norms, axis=1)
            if all(inclusions >= 0.):
                if faces[k] >= 0:
                    print("UV coordinate ", uv, " was on multiple faces.")
                faces[k] = face_i
        if faces[k] < 0:
            print("Couldn't convert UV coordinate ", uv, " that wasn't on known face.")
            continue
        face = mesh.faces[faces[k]]
        face_uvs = np.vstack([mesh.visual.uv[i, :] for i in face])
        face_xyzs = np.vstack([mesh.vertices[i, :] for i in face])
        # form this uv coordinate as a linear combination of the first three face UVs
        # and then use that same linear combination of the face verts to get the
        # xyz position
        uv_dir_1 = (face_uvs[1, :] - face_uvs[0, :]) / np.linalg.norm([face_uvs[1, :] - face_uvs[0, :]])
        uv_dir_2 = (face_uvs[2, :] - face_uvs[0, :]) / np.linalg.norm([face_uvs[2, :] - face_uvs[0, :]])
        c1, c2 = np.dot(np.linalg.inv(np.vstack([uv_dir_1, uv_dir_2])), uv - face_uvs[0])
        xyz_dir_1 = (face_xyzs[1, :] - face_xyzs[0, :]) / np.linalg.norm([face_xyzs[1, :] - face_xyzs[0, :]])
        xyz_dir_2 = (face_xyzs[2, :] - face_xyzs[0, :]) / np.linalg.norm([face_xyzs[2, :] - face_xyzs[0, :]])
        xyzs[k, :] = xyz_dir_1*c1 + xyz_dir_2*c2 + face_xyzs[0, :]
        normals[k, :] = mesh.vertex_normals[face[0]]
    return xyzs, normals, faces


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


def generate_perlin_noise_2d(shape, res):
    # from https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


if __name__ == "__main__":

    # attach to logger so trimesh messages will be printed to console
    # trimesh.util.attach_to_log()

    sx, sy, sz = np.random.uniform(low=0.1, high=0.25, size=(3,))
    print("Box shape: ", sx, sy, sz)
    mesh, box_uvs_by_face, box_uv_scale_factor, box_verts_by_face, box_normals_by_face = generate_scaled_box_with_uvs(sx, sy, sz)
    print("Box uv scale factor: ", box_uv_scale_factor)

    texture_size = 512
    baseColorTexture = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    metallicRoughnessTexture = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    metallicRoughnessTexture[..., -1] = 255

    # Fill the base color map with the cardboard texture
    cardboard_texture = np.array(PIL.Image.open("/home/gizatt/data/cardboard_box_texturing/textures/cardboard_tileable_1.png"))[:, :, :4]
    tile_box_with_texture(baseColorTexture, cardboard_texture, np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                          scale=[0.1, 0.1])
    #cardboard_texture = np.array(PIL.Image.open("/home/gizatt/data/cardboard_box_texturing/textures/maxresdefault.jpg"))
    #cardboard_texture = np.concatenate([cardboard_texture, 255*np.ones([cardboard_texture.shape[0], cardboard_texture.shape[1], 1])], axis=2)
    #cardboard_texture = np.array(PIL.Image.open("/home/gizatt/Downloads/grid.jpg"))
    #tile_box_with_texture(baseColorTexture, cardboard_texture, np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
    #                      number_of_tiles=[1, 1])
    
    LabelGenInfo = namedtuple('label_generation_info', field_names=[
        'type', # A key from type_to_path
        'faces', # list of string from [px, nx, py, ny, pz, nz]
        'uv_sampler', # callable to sample uv location on face
        'rotation_sampler', # callable to sample rotation,
        'roughness_sampler', # callable to sample rougness on [0., 1.]
        'width_in_meters', # float
        'occurance_prob_per_face' # 0 to 1 float or list of floats
    ])

    type_to_path = {
        'bar_code_printed': '/home/gizatt/data/cardboard_box_texturing/textures/bar_code_decal_1.png',
        'bar_code_sticker': '/home/gizatt/data/cardboard_box_texturing/textures/bar_code_sticker_decal_1.png',
        'recycleable_printed': '/home/gizatt/data/cardboard_box_texturing/textures/recycleable_decal.png',
        'sticker_bounds_printed': '/home/gizatt/data/cardboard_box_texturing/textures/sticker_placement_decal.png'
    }

    def random_mostly_on_face():
        return np.random.uniform([0.05, 0.05], [0.95, 0.95])
    def random_axis_aligned_rotation():
        return np.random.randint(4)*np.pi/2. + np.random.randn()*0.025
    def random_interval_factory(lb, ub):
        def random_internal():
            return np.random.uniform(lb, ub)
        return random_internal

    possible_labels_pre_tape = [
        LabelGenInfo(
            type='bar_code_printed', faces=['py', 'ny', 'px', 'nx', 'pz', 'nz'],
            uv_sampler = random_mostly_on_face,
            rotation_sampler = random_axis_aligned_rotation,
            roughness_sampler = random_interval_factory(0.5, 1.),
            width_in_meters=.04, occurance_prob_per_face=1.0),
        LabelGenInfo(
            type='bar_code_printed', faces=['py', 'ny', 'px', 'nx', 'pz', 'nz'],
            uv_sampler = random_mostly_on_face,
            rotation_sampler = random_axis_aligned_rotation,
            roughness_sampler = random_interval_factory(0.5, 1.),
            width_in_meters=.04, occurance_prob_per_face=0.5),
        LabelGenInfo(
            type='sticker_bounds_printed', faces=['py', 'ny', 'px', 'nx', 'pz', 'nz'],
            uv_sampler = random_mostly_on_face,
            rotation_sampler = random_axis_aligned_rotation,
            roughness_sampler = random_interval_factory(0.5, 1.),
            width_in_meters=.05, occurance_prob_per_face=1.0),
        LabelGenInfo(
            type='sticker_bounds_printed', faces=['py', 'ny', 'px', 'nx', 'pz', 'nz'],
            uv_sampler = random_mostly_on_face,
            rotation_sampler = random_axis_aligned_rotation,
            roughness_sampler = random_interval_factory(0.5, 1.),
            width_in_meters=.05, occurance_prob_per_face=1.0),
        LabelGenInfo(
            type='recycleable_printed', faces=['py', 'ny', 'px', 'nx', 'pz', 'nz'],
            uv_sampler = random_mostly_on_face,
            rotation_sampler = random_axis_aligned_rotation,
            roughness_sampler = random_interval_factory(0.5, 1.),
            width_in_meters=.04, occurance_prob_per_face=1.0),
    ]

    possible_labels_post_tape = [
        LabelGenInfo(
            type='bar_code_sticker', faces=[random.choice(['py', 'ny', 'px', 'nx', 'pz', 'nz'])],
            uv_sampler = random_mostly_on_face,
            rotation_sampler = random_axis_aligned_rotation,
            roughness_sampler = random_interval_factory(0.0, 1.),
            width_in_meters=.04, occurance_prob_per_face=1.0)
    ]

    all_applied_stickers = []
    def apply_sticker(destination_texture, sticker_texture, width, rotation, uv_loc, face):
        u_scale = width / box_uv_scale_factor
        v_scale = u_scale * sticker_texture.shape[1] / sticker_texture.shape[0]
        corners_pre_rotation = np.array([[-u_scale/2., -v_scale/2.],
                                          [u_scale/2., -v_scale/2.],
                                          [u_scale/2., v_scale/2.],
                                          [-u_scale/2., v_scale/2.]])
        rotmat = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        uv_bounds = box_uvs_by_face[face]
        offset = uv_loc * (np.max(uv_bounds, axis=0) - np.min(uv_bounds, axis=0)) + np.min(uv_bounds, axis=0)
        offset = np.array([offset[1], offset[0]]) # why???

        corners = np.dot(rotmat, corners_pre_rotation.T).T + offset
        scaling = np.ones(2) * (destination_texture.shape[1] * v_scale) / sticker_texture.shape[1]
        tile_box_with_texture(destination_texture, sticker_texture, corners, scale=scaling,
                              number_of_tiles=[1, 1])
        
        # Calculate world frame corners and normal, too
        world_corners, world_normals, world_faces = convert_mesh_uv_to_xyz(mesh, np.vstack([corners, np.mean(corners, axis=0)]))
        return world_corners, world_normals, world_faces

    def handle_label(label):
        for k, face in enumerate(label.faces):
            if isinstance(label.occurance_prob_per_face, list):
                occurance_prob_for_this_face = label.occurance_prob_per_face[k]
            else:
                occurance_prob_for_this_face = label.occurance_prob_per_face
            if np.random.random() < occurance_prob_for_this_face:
                rotation = label.rotation_sampler()
                uv_loc = label.uv_sampler()
                print("Applying label of type %s on %s at %f, %f" % (label.type, face, uv_loc[0], uv_loc[1]))
                assert(label.type in type_to_path.keys())
                sticker_texture = np.array(PIL.Image.open(type_to_path[label.type]))
                sticker_corners, sticker_normals, sticker_faces = apply_sticker(baseColorTexture, sticker_texture, label.width_in_meters, rotation, uv_loc, face)
                roughness_im = sticker_texture * 0
                roughness_im[sticker_texture[..., -1] > 0] = int(label.roughness_sampler() * 255)
                apply_sticker(metallicRoughnessTexture, roughness_im, label.width_in_meters, rotation, uv_loc, face)

                # And record that we applied it
                all_applied_stickers.append({
                    'type': label.type,
                    'corners_xyz': sticker_corners[:4, :].tolist(),
                    'corners_normal': sticker_normals[:4, :].tolist(),
                    'corners_faces': sticker_faces[:4].tolist(),
                    'center_xyz': sticker_corners[-1, :].tolist(),
                    'center_normal': sticker_normals[-1, :].tolist(),
                    'center_face': sticker_faces[-1].tolist()
                })
                
    
    for label in possible_labels_pre_tape:
        handle_label(label)

    # Write some random things on the box
    # Ref https://pythonprogramming.altervista.org/make-an-image-with-text-with-python/
    for k in range(2):
        face = random.choice(["px", "nx", "py", "nx", "pz", "nz"])
        string_length = np.random.randint(10) + 2
        text = ''.join(random.choice(string.ascii_letters + string.digits)
                       for i in range(string_length))
        # Render text to image
        font_size = 100
        fnt = PIL.ImageFont.truetype(
            random.choice(['arial.ttf', 'comic.ttf']), font_size)
        text_image = PIL.Image.new(mode = "RGBA", size = (int(font_size/2)*len(text),font_size+50))
        draw = PIL.ImageDraw.Draw(text_image)
        # draw text
        text_color = random.choice([
            (0, 0, 0),
            tuple(np.random.randint(255, size=3).tolist())])
        draw.text((10,10), text, font=fnt, fill=text_color)
        width = np.random.uniform(0.01, 0.05)
        rotation = random_axis_aligned_rotation()
        uv_loc = random_mostly_on_face()
        apply_sticker(baseColorTexture,
                      np.array(text_image), width=width, rotation=rotation, uv_loc=uv_loc, face=face)
        roughness_im = np.array(text_image, dtype=np.uint8) * 0
        roughness_im[np.sum(roughness_im, axis=2)] = np.random.randint(255)
        apply_sticker(metallicRoughnessTexture,
                      roughness_im, width=width, rotation=rotation, uv_loc=uv_loc, face=face)


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

        roughness = np.random.randint(255)
        tile_box_with_texture(metallicRoughnessTexture, np.ones((10, 10, 4))*roughness, corners)

    for k in range(2):
        # Draw more text with white background
        face = random.choice(["px", "nx", "py", "nx", "pz", "nz"])
        string_length = np.random.randint(10) + 2
        text = ''.join(random.choice(string.ascii_letters + string.digits)
                       for i in range(string_length))
        # Render text to image
        font_size = 100
        fnt = PIL.ImageFont.truetype(
            random.choice(['arial.ttf', 'comic.ttf']), font_size)
        text_image = PIL.Image.new("RGBA", (int(font_size/2)*len(text),font_size+50), (255, 255, 255, 255))
        draw = PIL.ImageDraw.Draw(text_image)
        # draw text
        text_color = random.choice([
            (0, 0, 0),
            tuple(np.random.randint(255, size=3).tolist())])
        draw.text((10,10), text, font=fnt, fill=text_color)
        width = np.random.uniform(0.01, 0.05)
        rotation = random_axis_aligned_rotation()
        uv_loc = random_mostly_on_face()
        apply_sticker(baseColorTexture,
                      np.array(text_image), width=width, rotation=rotation, uv_loc=uv_loc, face=face)
        roughness_im = np.array(text_image) * 0
        roughness_im[np.sum(roughness_im, axis=2)] = np.random.randint(255)
        apply_sticker(metallicRoughnessTexture,
                      roughness_im, width=width, rotation=rotation, uv_loc=uv_loc, face=face)

    for label in possible_labels_post_tape:
        handle_label(label)


    # Draw the completed texture, with the UV map overlayed
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(baseColorTexture)
    for k, face in enumerate(mesh.faces):
        uvs = np.vstack([mesh.visual.uv[v] for v in face])*baseColorTexture.shape[:2]
        patch = patches.Polygon(
            uvs, fill=False, linewidth=1.0, linestyle="--",
            edgecolor=plt.cm.jet(float(k) / len(mesh.faces)))
        plt.gca().add_patch(patch)
    plt.subplot(3, 1, 2)
    metallicRoughnessTexture = 255 - metallicRoughnessTexture[:, :, 0]
    plt.imshow(metallicRoughnessTexture)
    plt.subplot(3, 1, 3)

    normal_rotation_scale = np.random.uniform(0., np.pi/4)
    res_factor = 2**np.random.randint(10)
    normalTexture = np.stack(
        [generate_perlin_noise_2d(
        shape=(texture_size, texture_size),
        res=(int(texture_size/res_factor), int(texture_size/res_factor))) for k in range(2)],
        axis=-1) * normal_rotation_scale
    # In each channel, represents the rotation of 
    normalTexture = np.stack(
        [
            np.sin(normalTexture[:, :, 0]) * np.cos(normalTexture[:, :, 1]),
            np.sin(normalTexture[:, :, 0]) * np.sin(normalTexture[:, :, 1]),
            np.cos(normalTexture[:, :, 0])
        ], axis=2
    )
    normalTexture = (normalTexture*255/2. + 255/2.).astype(np.uint8)
    plt.imshow(normalTexture)
    #plt.show()
    plt.pause(0.5)

    out_dir = os.path.join("cardboard_boxes", "box_%04d" % np.random.randint(10000))
    os.system("mkdir -p %s" % out_dir)

    info_dict = {
        'scale': [float(sx), float(sy), float(sz)],
        'all_applied_stickers': all_applied_stickers
    }
    with open(os.path.join(out_dir, "info.yaml"), "w") as f:
        f.write(yaml.dump(info_dict))

    baseColorTexture = PIL.Image.fromarray(np.flipud(baseColorTexture))
    metallicRoughnessTexture = PIL.Image.fromarray(np.flipud(metallicRoughnessTexture))
    normalTexture = PIL.Image.fromarray(normalTexture)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=baseColorTexture,
        metallicRoughnessTexture=metallicRoughnessTexture,
        normalTexture=normalTexture)

    def save_im_as_jpg(im, path):
        if im.mode in ('RGBA', 'LA'):
            fill_color = 'black'
            background = PIL.Image.new(im.mode[:-1], im.size, fill_color)
            background.paste(im, im.split()[-1])
            im = background
        im.save(path)

    # Save out the generated box and textures
    with open(os.path.join(out_dir, "box.obj"), "w") as f:
        f.write(export_obj(mesh))
    save_im_as_jpg(baseColorTexture, os.path.join(out_dir, "box_col.jpg"))
    save_im_as_jpg(metallicRoughnessTexture, os.path.join(out_dir, "box_rgh.jpg"))
    save_im_as_jpg(normalTexture, os.path.join(out_dir, "box_nrm.jpg"))

    scene = trimesh.scene.scene.Scene()
    scene.add_geometry(mesh)
    trimesh.scene.lighting.autolight(scene)
    scene.show()
