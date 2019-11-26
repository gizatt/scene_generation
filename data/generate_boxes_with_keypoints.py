import argparse
from collections import namedtuple
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import time
import yaml
import sys

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import pydrake
from pydrake.systems.analysis import Simulator
from pydrake.common.eigen_geometry import Quaternion, AngleAxis, Isometry3
from pydrake.geometry import (
    Box,
    HalfSpace,
    SceneGraph,
    Sphere
)
from pydrake.geometry.render import (
    CameraProperties, DepthCameraProperties,
    MakeRenderEngineVtk, RenderEngineVtkParams
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import (
    SpatialInertia,
    UniformGravityFieldElement,
    UnitInertia
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant
)
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.systems.framework import BasicVector, DiagramBuilder, LeafSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.sensors import RgbdSensor

BoxWithLabel = namedtuple('BoxWithLabel', [
    'pose', # quatxyz
    'dimensions', # width (x), depth (y), height (z)
    'label_face', # "[pn][xyz]" e.g. "px" or "nz"
    'label_uv']) # uv coords, 0 to 1, on the given face of the label origin

def get_label_info_from_box_with_label(box):
    label_origin = np.zeros(3)
    s = 1
    if box.label_face[0] == 'n':
        s = -1
    box_label_dim = ord(box.label_face[1]) - ord('x')
    label_origin[box_label_dim] = s
    # In the other dims, offset by UV
    other_dims = [0, 1, 2]
    other_dims.remove(box_label_dim)
    label_origin[other_dims[0]] = (box.label_uv[0] - 0.5)*2.0
    label_origin[other_dims[1]] = (box.label_uv[1] - 0.5)*2.0
    label_origin *= box.dimensions/2.
    label_size = np.ones(3) * 0.05
    label_size[box_label_dim] = 0.001
    return label_origin, label_size

def generate_keypoints_from_box_with_label(box):
    assert(isinstance(box, BoxWithLabel))
    pts = np.array([[-1., -1., -1., -1, 1., 1., 1., 1.],
                    [-1., -1., 1., 1., -1., -1., 1., 1.],
                    [-1., 1., -1., 1., -1., 1., -1., 1.]])
    pts = (pts.T * box.dimensions).T / 2.
    # At make a point for the label origin
    label_origin, _ = get_label_info_from_box_with_label(box)
    pts = np.hstack([pts, label_origin[:, np.newaxis]])
    vals = np.zeros(pts.shape[1]).reshape(1, -1)
    vals[0, -1] = 1.
    quat = box.pose[:4] / np.linalg.norm(box.pose[:4])
    pts = RigidTransform(p=box.pose[-3:], quaternion=Quaternion(quat)).multiply(pts)
    return pts, vals

def generate_mbp_sg_diagram(seed):
    np.random.seed(seed)

    # Build up a list of boxes with sizes and label placements
    # TODO(gizatt): turn this into a yaml env, somehow?
    n_boxes = np.random.geometric(0.3) + 1
    boxes = []
    for box_i in range(n_boxes):
        # Generate random pose
        xyz = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.1, 0.5)])
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)
        pose = np.hstack([quat, xyz])
        # Random dimensions
        dimensions = np.array([
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.1, 0.3)
        ])
        label_face = "px" #np.random.choice(['p', 'n']) + \
                     #np.random.choice(['x', 'y', 'z'])
        label_uv = np.array([
            np.random.uniform(0.2, 0.8),
            np.random.uniform(0.2, 0.8)])

        boxes.append(
            BoxWithLabel(pose=pose, dimensions=dimensions,
                         label_face=label_face, label_uv=label_uv))


    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(
        builder, MultibodyPlant(time_step=0.005))

    # Add ground
    world_body = mbp.world_body()
    ground_shape = Box(4., 4., 1.)
    ground_body = mbp.AddRigidBody("ground", SpatialInertia(
        mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    mbp.WeldFrames(world_body.body_frame(), ground_body.body_frame(),
                   RigidTransform(p=[0, 0, -0.5]))
    mbp.RegisterVisualGeometry(
        ground_body, RigidTransform.Identity(), ground_shape, "ground_vis",
        np.array([0.5, 0.5, 0.5, 1.]))
    mbp.RegisterCollisionGeometry(
        ground_body, RigidTransform.Identity(), ground_shape, "ground_col",
        CoulombFriction(0.9, 0.8))

    for i, box in enumerate(boxes):
        body = mbp.AddRigidBody("box_{}".format(i), SpatialInertia(
            mass=0.1, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(0.01, 0.01, 0.01)))
        body_box = Box(*box.dimensions)
        mbp.RegisterVisualGeometry(
            body, RigidTransform.Identity(), body_box, "box_{}_vis".format(i),
            np.array([np.random.uniform(0.75, 0.95),
                      np.random.uniform(0.45, 0.55),
                      np.random.uniform(0.1, 0.2), 1.])) # random color
        mbp.RegisterCollisionGeometry(
            body, RigidTransform.Identity(), body_box, "box_{}_col".format(i),
            CoulombFriction(0.9, 0.8))

        # Draw a tiny thin box representing the label
        label_origin, label_size = get_label_info_from_box_with_label(box)
        label_box = Box(*label_size)
        mbp.RegisterVisualGeometry(
            body,
            RigidTransform(p=label_origin),
            label_box, "box_{}_vis_label".format(i),
            np.array([1., 0., 0., 1.]))        
    q0 = np.hstack([box.pose for box in boxes])
    mbp.Finalize()

    return builder, mbp, scene_graph, q0, boxes

def project_to_feasibility(mbp, mbp_context, q0, boxes):
    # Project to feasibility
    mbp.SetPositions(mbp_context, q0)
    ik = InverseKinematics(mbp, mbp_context)
    q_dec = ik.q()
    prog = ik.prog()
    def squaredNorm(x):
        return np.array([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2])
    for k in range(len(boxes)):
        # Quaternion norm
        prog.AddConstraint(
            squaredNorm, [1], [1], q_dec[(k*7):(k*7+4)])
        # Trivial quaternion bounds
        prog.AddBoundingBoxConstraint(
            -np.ones(4), np.ones(4), q_dec[(k*7):(k*7+4)])
        # Conservative bounds on on XYZ
        prog.AddBoundingBoxConstraint(
            np.array([-2., -2., -2.]), np.array([2., 2., 2.]),
            q_dec[(k*7+4):(k*7+7)])
    constraint = ik.AddMinimumDistanceConstraint(0.001)
    prog.AddQuadraticErrorCost(np.eye(q0.shape[0])*1.0, q0, q_dec)
    prog.SetInitialGuess(q_dec, q0)
    print("Projecting...")
    result = Solve(prog)
    print("Projected to feasibility with ", result.get_solver_id().name())
    print("\twith info ", result.get_solution_result())
    return result.GetSolution(q_dec)
    

def sample_and_draw_drake():
    # Generate a random collection of boxes
    seed = time.time()*1000*1000
    seed = int(seed % (2**32 - 1))
    builder, mbp, scene_graph, q0, boxes = generate_mbp_sg_diagram(seed=seed)

    visualizer = builder.AddSystem(MeshcatVisualizer(
        scene_graph,
        zmq_url="tcp://127.0.0.1:6000",
        draw_period=0.01))
    builder.Connect(scene_graph.get_pose_bundle_output_port(),
                    visualizer.get_input_port(0))

    # Add depth camera
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    width = 640
    height = 480
    color_properties = CameraProperties(
        width=width, height=height, fov_y=np.pi/2,
        renderer_name="vtk")
    depth_properties = DepthCameraProperties(
        width=width, height=height, fov_y=np.pi/2,
        renderer_name="vtk", z_near=0.1, z_far=5.5)

    # Put it at the origin.
    X_WB = RigidTransform(p=[0., 0., 1.0], rpy=RollPitchYaw(np.pi, 0., 0.))
    # This id would fail if we tried to render; no such id exists.
    parent_id = scene_graph.world_frame_id()
    camera_poses = RgbdSensor.CameraPoses(
        X_BC=RigidTransform(), X_BD=RigidTransform())
    sensor = builder.AddSystem(
        RgbdSensor(parent_id=parent_id, X_PB=X_WB,
                   color_properties=color_properties,
                   depth_properties=depth_properties,
                   camera_poses=camera_poses,
                   show_window=False))
    builder.Connect(scene_graph.get_query_output_port(),
                    sensor.query_object_input_port())

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    mbp_context = diagram.GetMutableSubsystemContext(
        mbp, diagram_context)
    sg_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_context)

    q0 = project_to_feasibility(mbp, mbp_context, q0, boxes)
    mbp.SetPositions(mbp_context, q0)

    # Simulate
    simulator = Simulator(diagram, diagram_context)
    #simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    simulator.StepTo(5.0)
    qf = mbp.GetPositions(mbp_context).copy()
    # Final pose is statically stable, by assumption.

    # Update box list poses
    for k, box in enumerate(boxes):
        box.pose[:] = qf[(k*7):((k+1)*7)]

    # Extract final depth image
    sensor_context = diagram.GetMutableSubsystemContext(
        sensor, diagram_context)
    depth_im = sensor.depth_image_32F_output_port().Eval(sensor_context)
    depth_im = np.frombuffer(depth_im.data, dtype=np.float32).reshape(depth_im.shape[:2])

    # Use the depth extrinsics and intrinsics to render the keypoints
    keypoints, vals = zip(*[generate_keypoints_from_box_with_label(box) for box in boxes])
    keypoints = np.hstack(keypoints)
    vals = np.hstack(vals)
    keypoints_in_cam = X_WB.inverse().multiply(keypoints)
    depth_K = sensor.depth_camera_info().intrinsic_matrix()
    keypoints_rendered = depth_K.dot(keypoints_in_cam)
    keypoints_rendered /= keypoints_rendered[2, :]

    # Prune out the keypoints based on their depth visibility
    # depths = np.linalg.norm(keypoints_in_cam, axis=0)
    # Depth is just camera Z level for this camera type
    depths = keypoints_in_cam[2, :]
    keep = np.zeros(keypoints.shape[1]).astype(int)
    for k in range(keypoints.shape[1]):
        v, u = np.round(keypoints_rendered[:2, k]).astype(int)
        if u < 0 or u >= 480 or v < 0 or v >= 640:
            continue
        if depths[k] < depth_im[u, v] + 1E-2:
            keep[k] = 1.
    keypoints_rendered_visible = keypoints_rendered[:, keep > 0]
    vals_visible = vals[:, keep > 0]

    #plt.figure()
    if plt.gcf():
        plt.gca().clear()
    plt.imshow(depth_im, cmap='summer')
    plt.scatter(keypoints_rendered_visible[0, :],
                keypoints_rendered_visible[1, :],
                c=vals_visible[0, :], cmap="winter")
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_xaxis()

    # Save out the environment info and keypoint list
    output_dict = {
        "n_objects": len(boxes)
    }
    for k, box in enumerate(boxes):
        ind_range = range(k*9,(k*9)+9)
        these_keypoints = keypoints[:, ind_range]
        these_vals = vals[:, ind_range]
        these_keep = keep[ind_range] > 0
        observed_keypoints = these_keypoints[:, these_keep]
        observed_vals = these_vals[:, these_keep]
        print("Saving observed keypoints and vals: ", observed_keypoints, observed_vals)
        output_dict["obj_%04d" % k] = {
            "class": "amazon_box",
            "pose": box.pose.tolist(),
            "dimensions": box.dimensions.tolist(),
            "label_face": box.label_face,
            "label_uv": box.label_uv.tolist(),
            "observed_keypoints": pickle.dumps(observed_keypoints),
            "observed_vals": pickle.dumps(observed_vals)
        }
    with open("box_observations.yaml", "a") as file:
        yaml.dump({"env_%d" % (round(time.time() * 1000)):
                   output_dict}, file)

    #print("Keypoints and vals in global frame: ", keypoints, vals)
    keypoints_and_vals_single = np.vstack([keypoints, vals])[:, :9]
    #np.savetxt("keypoints_obs.np", keypoints_and_vals_single[:, keep[:9] > 0])
    plt.pause(0.5)


if __name__ == "__main__":
    for k in range(50):
        sample_and_draw_drake()
    plt.show()