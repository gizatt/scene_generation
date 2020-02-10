import numpy as np
import collections

from trimesh import util

from trimesh.visual.color import to_float
from trimesh.visual.texture import TextureVisuals


def export_obj(mesh,
               include_normals=True,
               include_color=True,
               include_texture=True):
    """
    Export a mesh as a Wavefront OBJ file
    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be exported
    Returns
    -----------
    export : str
      OBJ format output
    """
    # store the multiple options for formatting
    # vertex indexes for faces
    face_formats = {('v',): '{}',
                    ('v', 'vn'): '{}//{}',
                    ('v', 'vt'): '{}/{}',
                    ('v', 'vn', 'vt'): '{}/{}/{}'}
    # we are going to reference face_formats with this
    face_type = ['v']

    # OBJ includes vertex color as RGB elements on the same line
    if include_color and mesh.visual.kind in ['vertex', 'face']:
        # create a stacked blob with position and color
        v_blob = np.column_stack((
            mesh.vertices,
            to_float(mesh.visual.vertex_colors[:, :3])))
    else:
        # otherwise just export vertices
        v_blob = mesh.vertices

    # add the first vertex key and convert the array
    export = 'v ' + util.array_to_string(v_blob,
                                         col_delim=' ',
                                         row_delim='\nv ',
                                         digits=8) + '\n'

    # only include vertex normals if they're already stored
    if include_normals and 'vertex_normals' in mesh._cache:
        # if vertex normals are stored in cache export them
        face_type.append('vn')
        export += 'vn '
        export += util.array_to_string(mesh.vertex_normals,
                                       col_delim=' ',
                                       row_delim='\nvn ',
                                       digits=8) + '\n'

    if include_texture:
      # Check mesh uv coords are the right shape
      assert(mesh.visual.uv.shape[0] == mesh.vertices.shape[0])
      face_type.append('vt')
      export += 'vt '
      export += util.array_to_string(mesh.visual.uv,
                                     col_delim=' ',
                                     row_delim='\nvt ',
                                     digits=8) + '\n'

    """
    TODO: update this to use TextureVisuals
    if include_texture:
        # if vertex texture exists and is the right shape export here
        face_type.append('vt')
        export += 'vt '
        export += util.array_to_string(mesh.metadata['vertex_texture'],
                                       col_delim=' ',
                                       row_delim='\nvt ',
                                       digits=8) + '\n'
    """

    # the format for a single vertex reference of a face
    face_format = face_formats[tuple(face_type)]
    faces = 'f ' + util.array_to_string(mesh.faces + 1,
                                        col_delim=' ',
                                        row_delim='\nf ',
                                        value_format=face_format)
    # add the exported faces to the export
    export += faces

    return export
