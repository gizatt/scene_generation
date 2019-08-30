import lxml.etree as et
import numpy as np
import trimesh
from trimesh.constants import log
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate

import os

import numpy as np


def export_urdf(mesh,
                directory,
                robot_name="robot",
                scale=1.0,
                color=[0.75, 0.75, 0.75],
                **kwargs):
    """
    Convert a Trimesh object into a URDF package for physics simulation.
    This breaks the mesh into convex pieces and writes them to the same
    directory as the .urdf file.

    Modified from Trimesh::exchange::urdf.py to fix inertias.

    Parameters
    ---------
    mesh      : Trimesh object
    directory : str
                  The directory path for the URDF package
    Returns
    ---------
    mesh : Trimesh object
             Multi-body mesh containing convex decomposition
    """

    # Extract the save directory and the file name
    fullpath = os.path.abspath(directory)
    name = os.path.basename(fullpath)
    _, ext = os.path.splitext(name)

    if ext != '':
        raise ValueError('URDF path must be a directory!')

    # Create directory if needed
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    elif not os.path.isdir(fullpath):
        raise ValueError('URDF path must be a directory!')

    # Perform a convex decomposition
    try:
        convex_pieces = trimesh.decomposition.convex_decomposition(
            mesh, **kwargs)
        if not isinstance(convex_pieces, list):
            convex_pieces = [convex_pieces]
    except BaseException:
        log.error('problem with convex decomposition, using hull',
                  exc_info=True)
        convex_pieces = [mesh.convex_hull]

    # open an XML tree
    root = et.Element('robot', name=robot_name)

    # Make this primary link
    link_name = "{}_body_link".format(robot_name)
    # Write the link out to the XML Tree
    link = et.SubElement(root, 'link', name=link_name)
    # Inertial information
    inertial = et.SubElement(link, 'inertial')
    et.SubElement(inertial, 'origin',
                  xyz="{:.2E} {:.2E} {:.2E}".format(
                        *mesh.center_mass.tolist()),
                  rpy="0 0 0")
    et.SubElement(inertial, 'mass', value='{:.2E}'.format(mesh.mass))
    I = [['{:.4E}'.format(y) for y in x]  # NOQA
         for x in mesh.moment_inertia]
    et.SubElement(
        inertial,
        'inertia',
        ixx=I[0][0],
        ixy=I[0][1],
        ixz=I[0][2],
        iyy=I[1][1],
        iyz=I[1][2],
        izz=I[2][2])

    # Add the original piece as visual geomeetry
    piece_name = '{}_visual'.format(name)
    piece_filename = '{}.obj'.format(piece_name)
    piece_filepath = os.path.join(fullpath, piece_filename)
    trimesh.exchange.export.export_mesh(mesh, piece_filepath)
    geom_name = '{}'.format(piece_filename)
    # Visual Information
    visual = et.SubElement(link, 'visual')
    et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
    geometry = et.SubElement(visual, 'geometry')
    et.SubElement(geometry, 'mesh', filename=geom_name,
                  scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                      scale,
                                                      scale))
    material = et.SubElement(visual, 'material', name='')
    if color == "random":
        this_color = trimesh.visual.random_color()
        mesh.visual.face_colors[:] = this_color
    else:
        this_color = color
        if len(this_color) == 3:
            this_color.append(1.)
        mesh.visual.face_colors[:] = np.array(this_color)*255
    et.SubElement(material,
                  'color',
                  rgba="{:.2E} {:.2E} {:.2E} {:.2E}".format(
                    this_color[0],
                    this_color[1],
                    this_color[2],
                    this_color[3]))    

    # Loop through all pieces, adding each as visual + collision geometry
    for i, piece in enumerate(convex_pieces):

        # Save each nearly convex mesh out to a file
        piece_name = '{}_convex_piece_{}'.format(name, i)
        piece_filename = '{}.obj'.format(piece_name)
        piece_filepath = os.path.join(fullpath, piece_filename)
        trimesh.exchange.export.export_mesh(piece, piece_filepath)

        geom_name = '{}'.format(piece_filename)

        # Visual Information
        #visual = et.SubElement(link, 'visual')
        #et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
        #geometry = et.SubElement(visual, 'geometry')
        #et.SubElement(geometry, 'mesh', filename=geom_name,
        #              scale="{:.4E} {:.4E} {:.4E}".format(scale,
        #                                                  scale,
        #                                                  scale))
        #material = et.SubElement(visual, 'material', name='')
        #if color == "random":
        #    this_color = trimesh.visual.random_color()
        #    piece.visual.face_colors[:] = this_color
        #else:
        #    this_color = color
        #    if len(this_color) == 3:
        #        this_color.append(1.)
        #    piece.visual.face_colors[:] = np.array(this_color)*255
        #et.SubElement(material,
        #              'color',
        #              rgba="{:.2E} {:.2E} {:.2E} {:.2E}".format(
        #                this_color[0],
        #                this_color[1],
        #                this_color[2],
        #                this_color[3]))
#
        # Collision Information
        collision = et.SubElement(link, 'collision')
        et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(collision, 'geometry')
        et.SubElement(geometry, 'mesh', filename=geom_name,
                      scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                          scale,
                                                          scale),
                      convex="True")

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = '{}.urdf'.format(name)
    tree.write(os.path.join(fullpath, urdf_filename),
               pretty_print=True)

    # Write Gazebo config file
    root = et.Element('model')
    model = et.SubElement(root, 'name')
    model.text = name
    version = et.SubElement(root, 'version')
    version.text = '1.0'
    sdf = et.SubElement(root, 'sdf', version='1.4')
    sdf.text = '{}.urdf'.format(name)

    author = et.SubElement(root, 'author')
    et.SubElement(author, 'name').text = 'trimesh {}'.format(trimesh.__version__)
    et.SubElement(author, 'email').text = 'blank@blank.blank'

    description = et.SubElement(root, 'description')
    description.text = name

    tree = et.ElementTree(root)
    tree.write(os.path.join(fullpath, 'model.config'))

    return convex_pieces


def do_generate_mug():
    # Mug base is a cylinder
    root_radius = 0.05
    root_thickness = 0.01
    root_height = 0.1
    root_sections = 20

    handle_radius = 0.005
    handle_start_height = 0.01
    handle_end_height = 0.09
    handle_wingspan = 0.04

    total_density = 1000 # kg / m^3

    root_mesh_outer = trimesh.primitives.Cylinder(
        radius=root_radius, height=root_height, sections=root_sections)

    inner_tf = np.eye(4)
    inner_tf[2, 3] = root_thickness
    root_mesh_inner = trimesh.primitives.Cylinder(
        radius=root_radius - root_thickness,
        height=root_height - root_thickness, sections=root_sections,
        transform=inner_tf)

    root_mesh = trimesh.boolean.difference([root_mesh_outer, root_mesh_inner])

    # Generate a handle by extruding a circle along a spline
    handle_path = np.vstack([[root_radius-handle_radius, 0., handle_start_height],
                             [root_radius + handle_wingspan, 0., handle_start_height*0.75 + handle_end_height*0.25],
                             [root_radius + handle_wingspan, 0., handle_start_height*0.25 + handle_end_height*0.75],
                             [root_radius-handle_radius, 0., handle_end_height]])
    handle_path[:, 2] -= root_height / 2.
    handle_mesh_base = trimesh.creation.sweep_polygon(
        polygon=Point([0, 0]).buffer(handle_radius),
        path=handle_path)
    trimesh.repair.fix_normals(handle_mesh_base)

    handle_mesh = trimesh.boolean.difference([
        handle_mesh_base, root_mesh_outer])

    mesh = trimesh.boolean.union([
        handle_mesh,
        root_mesh])
    mesh.density = total_density
    # preview mesh in an opengl window if you installed pyglet with pip
    #mesh.show()

    # For opts see `testVHACD --help`
    # Res 1E5 is default and works great.
    # 1E6 was a little cleaned but took 10x as long. (Glad it's only linear.)
    mesh_decomp = export_urdf(
        mesh, "mug", robot_name="mug",
        color="random", maxNumVerticesPerCH=64, pca=1,
        resolution=1E5)
    scene = trimesh.scene.scene.Scene()
    for part in mesh_decomp:
        scene.add_geometry(part)
    scene.show()


if __name__ == "__main__":

    # attach to logger so trimesh messages will be printed to console
    trimesh.util.attach_to_log()

    do_generate_mug()