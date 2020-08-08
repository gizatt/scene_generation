import numpy as np
import os
from PIL import Image
from pyrr import Matrix44
import moderngl
from objloader import Obj
import yaml
from scene_generation.utils.type_convert import (
    transform_from_pose_vector,
    dict_to_matrix
)
import matplotlib.cm as cm
import tqdm

def prep_program():
    ctx = moderngl.create_context(standalone=True, backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)

    prog = ctx.program(
        vertex_shader='''
            #version 330
            uniform mat4 Mvp;
            in vec3 in_position;
            in vec3 in_vert_local;
            out vec3 vert_local;
            void main() {
                gl_Position = Mvp * vec4(in_position, 1.0);
                vert_local = in_vert_local;
            }
        ''',
        fragment_shader='''
            #version 330
            uniform mat4 Mvp;
            uniform sampler2D colorMap;
            in vec3 vert_local;
            out vec4 f_color;
            void main() {
                float distance = sqrt((vert_local[0]*vert_local[0]+vert_local[1]*vert_local[1]+vert_local[2]*vert_local[2]))/1.414;
                f_color = texture(colorMap, vec2(distance, 0.));
            }
        ''',
    )

    # Pregenerate colormap from matplotlib
    colormap_data = cm.get_cmap('viridis')(np.linspace(0., 1., 1000))*255
    colormap_texture = ctx.texture((1000, 1), components=4, data=np.ascontiguousarray(colormap_data.astype(np.int8)))
    colormap_sampler = ctx.sampler(texture=colormap_texture)
    colormap_texture.use(location=0)
    prog['colorMap'] = 0

    return ctx, prog

def render_descriptor_map(scene_folder, ctx, prog):
    scene_info_path = os.path.join(scene_folder, "scene_info.yaml")
    with open(scene_info_path, 'r') as f:
        scene_info = yaml.load(f, Loader=yaml.FullLoader)
    for scene_k, scene_view in enumerate(scene_info["data"]):
        # Set up the scene
        verts_vbos = []
        vert_local_vbos = []
        for object_name in list(scene_view["object_poses"].keys()):
            object_sdf = scene_info["objects"][object_name]["sdf"]
            object_obj = base_data_path + object_sdf[:-4] + ".obj"
            object_tf = transform_from_pose_vector(np.array(scene_view["object_poses"][object_name]))
            obj = Obj.open(object_obj)
            verts_norms_uvs = obj.to_array().T # 9 x 36
            norms = verts_norms_uvs[3:6, :]
            verts = verts_norms_uvs[:3, :] # 3 x 36
            verts_local = (verts / np.abs(verts)) * (norms == 0.0)
            verts_global = np.dot(object_tf, np.vstack([verts, np.ones((1, verts.shape[1]))]))[:3, :]
            verts_vbos.append(ctx.buffer(np.ascontiguousarray(verts_global.astype(np.float32).T)))
            vert_local_vbos.append(
                ctx.buffer(np.ascontiguousarray(verts_local.astype(np.float32).T)))


        for camera_name in list(scene_view["camera_frames"].keys()):
            height = scene_info["cameras"][camera_name]["calibration"]["image_height"]
            width = scene_info["cameras"][camera_name]["calibration"]["image_width"]

            fbo = ctx.simple_framebuffer((width, height), components=4)
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 0.0)

            near = 0.1
            far = 10.0
            view_tf = transform_from_pose_vector(np.array(scene_view["camera_frames"][camera_name]["pose"]))
            orig_projection_matrix = dict_to_matrix(scene_info["cameras"][camera_name]["calibration"]["projection_matrix"])
            aspect = float(width) / float(height)
            fovy = 2. * np.arctan2(height,  2. * orig_projection_matrix[1, 1])
            proj = Matrix44.perspective_projection(fovy*180./np.pi, aspect, near, far)
            cam_origin = view_tf[:3, 3]
            cam_forward = view_tf[:3, :3].dot(np.array([0., 0., 1.]))
            cam_up = view_tf[:3, :3].dot(np.array([0., -1., 0.]))

            look_at = Matrix44.look_at(cam_origin,
                                       cam_origin+cam_forward,
                                       cam_up)
            prog['Mvp'].write((proj * look_at).astype(np.float32))

            for vert_vbo, vert_local_vbo in zip(verts_vbos, vert_local_vbos):
                vao = ctx.vertex_array(prog,
                    [
                        (vert_vbo, "3f", 'in_position'),
                        (vert_local_vbo, "3f", 'in_vert_local')
                    ])
                vao.render(mode=moderngl.TRIANGLES)

            # Output next to the RGB image with a postfix (like the many other label image types)
            filename = scene_view["camera_frames"][camera_name]["rgb_image_filename"]
            filename = filename[:-4] + "_descriptors.png"
            image = Image.frombytes('RGBA', (width, height), fbo.read(components=4))
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(os.path.join(scene_folder, filename),
                       format='png')

if __name__ == '__main__':
    ctx, prog = prep_program()

    scene_groups_to_render = [
        "scene_group_5_types_100",
        "scene_group_200",
        "scene_group_single_box_topdown_50",
        "scene_group_topdown_views_10",
        "scene_group_mini",
        "scene_group_topdown_views_50",
        "scene_group_topdown_views_50_2",
    ]
    base_data_path = "/home/gizatt/data/generated_cardboard_envs/"

    print("Collecting folders...")
    scene_folders_to_render = []
    for scene_group_folder in scene_groups_to_render:
        folder = os.path.join(base_data_path, scene_group_folder)
        if os.path.isdir(folder):
            for scene_folder in os.listdir(folder):
                subfolder = os.path.join(folder, scene_folder)
                scene_info_path = os.path.join(subfolder, "scene_info.yaml")
                if os.path.exists(scene_info_path):
                    scene_folders_to_render.append(subfolder)
    print("Rendering descriptor maps...")
    for subfolder in tqdm.tqdm(scene_folders_to_render):
        render_descriptor_map(subfolder, ctx, prog)