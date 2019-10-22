import lxml.etree as et
import skimage.draw
import skimage.transform
import matplotlib.pyplot as plt
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

def generate_unit_box_face_uvs():
    # Using the default UV unwrapping from blender for the unit cube, each face gets
    # projected like:
    # TODO(gizatt) Could probably retrieve these automatically by checking normals
    # or exporting my unit cube mesh with quads...
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

    # Rescale the declar first
    scaling = np.ones(3)
    scaling[:2] = scale
    decal_image_scaled = scipy.ndimage.zoom(decal_image, scaling, mode='wrap')

    number_of_tiles = (np.max(corners, axis=0) - np.min(corners, axis=0)) * \
        base_image.shape[:2] / decal_image_scaled.shape[:2]
    # Tile the decal image a sufficient number of times
    reps = np.hstack([np.ceil(number_of_tiles).astype(int), 1])
    print(reps)
    decal_full = np.tile(decal_image_scaled, reps=reps)
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

    mesh = trimesh.load_mesh('/home/gizatt/data/cardboard_box_texturing/unit_box.obj')
    
    # Rescale the box.
    scaling_amounts = np.array([1.0, 1.0, 0.5])
    before_verts = mesh.vertices.copy()
    for k in range(len(scaling_amounts)):
        mesh.vertices[:, k] *= scaling_amounts[k]

    # Forward UV mapping: x, y, z = M(u, v)
    # Spatial warping: xout, yout, zout = S(x, y, z)
    # We want to figure out, for every point in UV space, what the partial derivatives
    # in the U and V directions of the spatial warp function is.
    # By the chain rule, \grad_{uv} M^-1(S(xyz) = \grad_{uv}M^-1(S(xyz)) * \grad_{xyz} S(xyz)
    # i.e. hints that we should be projecting the local spatial warping Jacobian into the
    # local tangent space of the UV map.
    # (probably mixing my metaphors a bit, but maybe that gets the idea across).
    #
    # We can evaluate that for the verts of each triangle, knowing that in our case
    # \grad_{x,y,z}S(x,y,z) is the constant scale factor listed above, and build a map over
    # UV space of the U and V derivatives at each pixel.

    def compute_triangle_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(a - c, b - c))
    rescaling_by_face = []
    for face in mesh.faces:
        assert(len(face) == 3)
        # [x y z]^T = A [u v 1]^T
        # Solve with least squares, in case it's singular:
        uvstack = np.vstack([mesh.visual.uv[v] for v in face]).T
        uvstack = np.vstack([uvstack, np.ones(3)]).T
        xyzstack = np.vstack([before_verts[v] for v in face])
        A, _, _, _ = np.linalg.lstsq(uvstack, xyzstack)
        # TODO(gizatt): Why can't I just solve this problem the other way around?
        # I think because it's not well posed the other way around...?
        J_minv = np.linalg.pinv(A[:2, :])
        # Crush the zero entries
        J_minv[np.abs(J_minv) < 1E-6] = 0.
        # Normalize columns
        J_minv /= np.linalg.norm(J_minv, axis=0)

        # Project our scaling into that
        local_scaling = np.abs(scaling_amounts.dot(J_minv))
        rescaling_by_face.append(local_scaling)

    # Finally, assuming that the UV mapping is one-to-one, a uv point [ui, vi] in the
    # original UV map should correspond to a point [uo, vo] in a rescaled UV
    # map, where uo (or vo) is the integral from 0 to ui (or vi) of the local scaling
    # of the spatial surface area of the corresponding mesh faces.
    # We'll approximate those integrals by drawing the rescalings into pixels of a
    # 2D image and taking 1D cumulative sums in each direction.
    u_scaling_image = np.ones((1024, 1024))
    v_scaling_image = np.ones((1024, 1024))
    for face, rescaling in zip(mesh.faces, rescaling_by_face):
        # Get pixel region
        uvs_before = np.vstack([mesh.visual.uv[v] for v in face])*u_scaling_image.shape
        rr, cc = skimage.draw.polygon(uvs_before[:, 0], uvs_before[:, 1], u_scaling_image.shape)
        u_scaling_image[rr, cc] = rescaling[0]
        v_scaling_image[rr, cc] = rescaling[1]
    plt.subplot(5, 1, 1)
    plt.imshow(np.dstack([u_scaling_image, v_scaling_image, u_scaling_image*0.]))
    integral_image_u = np.cumsum(u_scaling_image, axis=0) / u_scaling_image.shape[0]
    integral_image_v = np.cumsum(v_scaling_image, axis=1) / v_scaling_image.shape[1]
    #unit_integral_image = skimage.transform.integral.integral_image(np.ones(u_scaling_image.shape))
    #integral_image_u = skimage.transform.integral.integral_image(u_scaling_image) / unit_integral_image
    #integral_image_v = skimage.transform.integral.integral_image(v_scaling_image) / unit_integral_image
    plt.subplot(5, 1, 2)
    plt.imshow(integral_image_u)
    plt.subplot(5, 1, 3)
    plt.imshow(integral_image_v)

    plt.subplot(5, 1, 4)
    uv_map_image = np.zeros((1024, 1024))
    for k, face in enumerate(mesh.faces):
        uvs = np.vstack([mesh.visual.uv[v] for v in face])*uv_map_image.shape
        rr, cc = skimage.draw.polygon(uvs[:, 0], uvs[:, 1], uv_map_image.shape)
        uv_map_image[rr, cc] = k + 1
    plt.imshow(uv_map_image)

    # Finally, rescale all of the uv coordinates in place using those mappings.
    for k, uv_i in enumerate(mesh.visual.uv):
        coords = np.round(uv_i * u_scaling_image.shape)
        coords = np.clip(coords, np.zeros(2), np.array(u_scaling_image.shape) - 1).astype(int)
        print("UV i: ", uv_i)
        print("Final coords: ", coords)
        mesh.visual.uv[k] = np.array([integral_image_u[coords[0], coords[1]],
                                      integral_image_v[coords[0], coords[1]]])

    uv_map_image = np.zeros((1024, 1024))
    for k, face in enumerate(mesh.faces):
        uvs = np.vstack([mesh.visual.uv[v] for v in face])*uv_map_image.shape
        rr, cc = skimage.draw.polygon(uvs[:, 0], uvs[:, 1], uv_map_image.shape)
        uv_map_image[rr, cc] = k + 1
    plt.subplot(5, 1, 5)
    plt.imshow(uv_map_image)
    plt.show()
    
    #out_image = np.zeros((1024, 1024))
    #Xi, Yi = np.meshgrid(range(out_image.shape[0]), range(out_image.shape[1]))
    #print(Xi, Yi)
    #print(Xi.shape)
    #interp = scipy.interpolate.RectBivariateSpline(range(out_image.shape[0]), range(out_image.shape[1]), uv_map_image)
    #Xo = out_image.shape[0] * integral_image_u
    #Yo = out_image.shape[1] * integral_image_v
    #warped_out_image = interp(Xo, Yo, grid=False)
    #plt.subplot(5, 1, 5)
    #plt.imshow(uv_map_image)
    #plt.show()

    print("Mesh verts: ", mesh.vertices)
    print("Mesh normals: ", mesh.vertex_normals)
    print("Mesh faces: ", mesh.faces)
    #print("Mesh visual verts: ", mesh.visual.vertices)
    #print("Mesh visual faces: ", mesh.visual.faces)
    print("Mesh visual UVs: ", mesh.visual.uv)

    # Generate a base texture for the box using a cardboard texture
    baseColorTexture = np.zeros((2048, 2048, 3), dtype=np.uint8)

    # Fill the base color map with the cardboard texture
    #cardboard_texture = np.array(PIL.Image.open("/home/gizatt/data/cardboard_box_texturing/textures/cardboard_tileable_1.png"))[:, :, :3]
    cardboard_texture = np.array(PIL.Image.open("/home/gizatt/Downloads/grid.jpg"))
    print(cardboard_texture)
    tile_box_with_texture(baseColorTexture, cardboard_texture, np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                          scale=[1., 1.], rotation=0.0)
    
    # Generate 


    baseColorTexture = PIL.Image.fromarray(baseColorTexture)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=baseColorTexture)
    mesh.show()