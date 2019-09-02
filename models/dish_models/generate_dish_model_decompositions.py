import sys
import trimesh
from scene_generation.models.generate_mugs import export_urdf

if len(sys.argv) != 2:
    print("Please say which model to regenerate (or `all`)")
    exit(0)

do_all = (sys.argv[1] == "all")
do_viz = True


def do_convex_decomposition_to_urdf(obj_filename, obj_mass, output_directory, do_visualization=False, scale=1.0, color=[0.75, 0.75, 0.75], **kwargs):
    mesh = trimesh.load(obj_filename)
    mesh.apply_scale(scale)  # applies physical property scaling

    if (do_visualization):
        print("Showing input mesh...")
        mesh.show()

    mesh.density = obj_mass / mesh.volume
    decomposed_mesh = export_urdf(mesh, output_directory, robot_name=output_directory, color=color, **kwargs)

    print("Input mesh had ", len(mesh.faces), " faces and ", len(mesh.vertices), " verts")
    print("Output mesh has ", sum([len(piece.faces) for piece in decomposed_mesh]),
          " faces and ", sum([len(piece.vertices) for piece in decomposed_mesh]), " verts")

    if (do_visualization):
        print("Showing output mesh...")
        scene = trimesh.scene.scene.Scene()
        for part in decomposed_mesh:
            scene.add_geometry(part)
        scene.show()

if do_all or sys.argv[1] == "plate_11in":
    do_convex_decomposition_to_urdf("meshes/visual/plate_11in.obj", \
        0.282, \
        "plate_11in_decomp", \
        do_visualization = do_viz, \
        scale=0.001, \
        color=[0.5, 0.9, 0.5], \
        **{"maxhulls": 16, "maxNumVerticesPerCH": 40})

if do_all or sys.argv[1] == "plate_8p5in":
    do_convex_decomposition_to_urdf("meshes/visual/plate_8p5in.obj", \
        0.180, \
        "plate_8p5in_decomp", \
        do_visualization = do_viz, \
        scale=0.001, \
        color=[0.9, 0.5, 0.5], \
        **{"maxhulls": 16, "maxNumVerticesPerCH": 40})

if do_all or sys.argv[1] == "bowl_6p25in":
    do_convex_decomposition_to_urdf("meshes/visual/bowl_6p25in.obj", \
        0.140, \
        "bowl_6p25in_decomp", \
        do_visualization = do_viz, \
        scale=0.001, \
        color=[0.5, 0.5, 0.9], \
        **{"maxhulls": 16, "maxNumVerticesPerCH": 40})

#if do_all or sys.argv[1] == "dish_rack":
#   do_convex_decomposition_to_urdf("meshes/visual/dish_rack.obj", \
#       0.140, \
#       "dish_rack_decomp", \
#       do_visualization = do_viz, \
#       scale=0.001, \
#       **{"maxhulls": 12, "maxNumVerticesPerCH": 15})