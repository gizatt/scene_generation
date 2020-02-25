## Dataset details

TODO: Move `data/generate_cardboard_boxes.py` and `generate_cardboard_box_environments.py` into this directory, along with their support material.

### Individual Synthetic Scene scene_info.yaml

scene_info.yaml formatting:
```
objects: // static map of scene-specific object name to object class info
   object_name:
      class: class_name
      sdf_path: relative path to the object SDF/URDF from the dataset root
      label_index: int
      keypoints: 4xN matrix dict of keypoints in body frame

cameras: // static map of camera names to calibration info
    camera_name:
        calibration: dict of matrix dicts of ROS-format camera intrinsics

data:  // one data entry per time step w/ same object poses but possibly >1 camera
    -   camera_frames:
            camera_name: 
                pose: quat xyz format pose vector
                rgb_image_filename: path
                depth_image_filename: path
                label_image_filename: path
        object_poses:
            object_name: quat xyz format pose vector
```

### Raw Synthetic Scene Database

The "raw" synthetic scene output format is a list of folders named like `scene_%03d`, where each scene contains the generated images from that scene. (This can include multiple time steps and multiple cameras.) A complete description of what happened in that scene is included in `scene_info.yaml`. For dataset management convenience, a complete raw synthetic scene database is organized like:

- cardboard_boxes (*contains the raw box objs, textures, SDF, and info yaml)
  - box_0001
  - box_0002
  - ...
- scene_group_1 (*contains scene subfolders*)
  - scene_001
  - scene_002
  - ...
- scene_group_2 (*contains scene subfolders*)
- ...
- scene_groups_train.txt (*raw text file, with one scene group folder name per line*)
- scene_groups_test.txt (*raw text file, with one scene group folder name per line*)

This raw collection is compiled into a more convenient linear (like COCO) high-level description file by `compile_generated_scenes_to_dataset.py`, which produces a single COCO-like JSON that comprehensively lists all of the training examples, source images, bbox information, camera calibration, etc for each annotation:
- "info": COCO-standard info
- "licenses": COCO-standard licenses
- "categories": Superset of COCO-standard list of categories. For this project, this is currently just "prime_box" with ID 1, plus a "parameters" field listing the name of each parameter in order (scale_x, scale_y, scale_z) and "pose" field listing pose params in (qw, qx, qy, qz, x, y, z) order.
- "images": Superset of COCO standard: list of dicts, one dict for each image, assigning each image a unique ID, but also listing the camera intrinsics and extrinsics for the camera that took the image.
- "annotations": List of annotation dicts, including the standard COCO entries, plus the vector-format of my special type of ambiguous keypoint observations (non-coco format since these are ambiguous keypoints), plus object-specific pose + shape information (as defined in the object category), plus the path to the root of that object's actual folder (with the obj + textures + stuff).
