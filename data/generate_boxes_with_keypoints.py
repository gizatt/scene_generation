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
    #plt.show()


def sample_and_draw_pyro():
    import torch
    import pyro
    import pyro.infer
    import pyro.optim
    import pyro.distributions as dist
    import torch.distributions.constraints as constraints

    def generate_single_box(name="box"):
        quat = pyro.sample(
            name + "_quat",
            dist.Uniform(torch.ones(4).double()*-1.01,
                         torch.ones(4).double()*1.01).to_event(1)
        )
        quat = quat / torch.norm(quat)

        xyz_mean = torch.zeros(3).double()
        xyz_scale = torch.ones(3).double() * 0.1
        xyz = pyro.sample(
            name + "_xyz",
            dist.Normal(xyz_mean, xyz_scale).to_event(1)
        )

        dimensions_alpha = pyro.param(
            "dimensions_alpha", torch.ones(3).double() * 3.,
            constraint=constraints.positive)
        dimensions_beta = pyro.param(
            "dimensions_beta", torch.ones(3).double() * 1.,
            constraint=constraints.positive)
        dimensions = pyro.sample(
            name + "_dimensions",
            dist.InverseGamma(dimensions_alpha, dimensions_beta).to_event(1)
        )

        face_weights = torch.ones(6).double() / 6.
        possible_labels = ['px', 'py', 'pz', 'nx', 'ny', 'nz']
        label_face_ind = pyro.sample(name + "_label_face",
            dist.Categorical(face_weights),
            infer={"enumerate":"sequential"})
        label_face = possible_labels[label_face_ind]

        # UV choices
        label_uv = pyro.sample(
            name + "_label_uv",
            dist.Uniform(torch.zeros(2).double(),
                         torch.ones(2).double()).to_event(1)
        )

        sampled_box = BoxWithLabel(
            pose = torch.cat([quat, xyz]),
            dimensions = dimensions,
            label_face = label_face,
            label_uv = label_uv)
        return sampled_box

    def get_label_info_from_box_with_label(box):
        label_origin = torch.zeros(3, dtype=box.dimensions.dtype)
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
        label_size = torch.ones(3, dtype=box.dimensions.dtype) * 0.05
        label_size[box_label_dim] = 0.001
        return label_origin, label_size

    # https://github.com/lkhphuc/pytorch-3d-point-cloud-generation/blob/master/transform.py
    def quaternionToRotMatrix(q):
        # q = [V, 4]
        qa, qb, qc, qd = torch.unbind(q, dim=1) # [V,]
        R = torch.stack(
            [torch.stack([1 - 2 * (qc**2 + qd**2),
                          2 * (qb * qc - qa * qd),
                          2 * (qa * qc + qb * qd)]),
             torch.stack([2 * (qb * qc + qa * qd),
                          1 - 2 * (qb**2 + qd**2),
                          2 * (qc * qd - qa * qb)]),
             torch.stack([2 * (qb * qd - qa * qc),
                          2 * (qa * qb + qc * qd),
                          1 - 2 * (qb**2 + qc**2)])]
        ).permute(2, 0, 1)
        return R.to(q.device)


    def rotMatrixToQuaternion(R):
        """
        R = [3, 3]
        From https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
        This code uses a modification of the algorithm described in:
        https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        which is itself based on the method described here:
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        Altered to work with the column vector convention instead of row vectors
        """
        q = torch.empty(4, dtype=R.dtype)
        R = torch.t(R)
        if R[2, 2] < 0:
            if R[0, 0] > R[1, 1]:
                t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
                q[:] = torch.tensor([R[1, 2]-R[2, 1],  t,  R[0, 1]+R[1, 0],  R[2, 0]+R[0, 2]])
            else:
                t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
                q[:] = torch.tensor([R[2, 0]-R[0, 2],  R[0, 1]+R[1, 0],  t,  R[1, 2]+R[2, 1]])
        else:
            if R[0, 0] < -R[1, 1]:
                t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
                q[:] = torch.tensor([R[0, 1]-R[1, 0],  R[2, 0]+R[0, 2],  R[1, 2]+R[2, 1],  t])
            else:
                t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
                q[:] = torch.tensor([t,  R[1, 2]-R[2, 1],  R[2, 0]-R[0, 2],  R[0, 1]-R[1, 0]])

        q = q * 0.5 / torch.sqrt(t)
        return q


    def transParamsToHomMatrix(q, t):
        """q = [V, 4], t = [V,3]"""
        N = q.size(0)
        R = quaternionToRotMatrix(q) # [V,3,3]
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=2) # [V,3,4]
        hom_aug = torch.cat([torch.zeros([N, 1, 3]).double(),
                             torch.ones([N, 1, 1]).double()],
                            dim=2).to(Rt.device)
        RtHom = torch.cat([Rt, hom_aug], dim=1) # [V,4,4]
        return RtHom

    def generate_keypoints_from_box_with_label(box):
        assert(isinstance(box, BoxWithLabel))
        pts = torch.tensor(
            np.array([[-1., -1., -1., -1, 1., 1., 1., 1.],
                      [-1., -1., 1., 1., -1., -1., 1., 1.],
                      [-1., 1., -1., 1., -1., 1., -1., 1.]])).double()
        pts = (pts.T * box.dimensions).T / 2.
        # At make a point for the label origin
        label_origin, _ = get_label_info_from_box_with_label(box)
        pts = torch.cat([pts, label_origin[:, np.newaxis]], dim=1)
        vals = torch.zeros(pts.shape[1]).double().reshape(1, -1)
        vals[0, -1] = 1.
        quat = box.pose[:4].reshape(1, -1)
        xyz = box.pose[-3:].reshape(1, -1)
        tf_mat = transParamsToHomMatrix(quat, xyz)[0]
        pts = torch.mm(tf_mat, torch.cat([pts, torch.ones(pts.shape[1]).double().reshape(1, -1)], dim=0))[:3, :]
        return pts, vals

    # Simple generative model of keypoints: assume a number of observed keypoints are
    # drawn from a GMM with components from each keypoint plus an "outlier" component.
    # The number of observed keypoints is drawn from an (uninformative) discrete
    # uniform distribution. (Maybe something with infinite support would be better?)
    def model_keypoint_observations_gmm(keypoints, vals):
        max_num_keypoints = keypoints.shape[1] + 5
        num_keypoints = pyro.sample(
            "num_observed_keypoints_minus_one",
            dist.Categorical(probs=torch.ones(max_num_keypoints).double())) + 1

        keypoint_var = pyro.param(
            "keypoint_var",
            torch.tensor([0.01]).double(),
            constraint=constraints.positive)
        val_var = pyro.param(
            "val_var",
            torch.tensor([0.01]).double(),
            constraint=constraints.positive)
        outlier_var = pyro.param(
            "outlier_var",
            torch.tensor([1.0]).double(),
            constraint=constraints.positive)
        outlier_weight = pyro.param(
            "outlier_weight",
            torch.tensor([0.01]).double(),
            constraint=constraints.positive)

        # Make the maodel components
        locs = torch.cat([keypoints, vals], dim=0)
        scales = torch.cat([torch.ones(keypoints.shape).double()*keypoint_var,
                            torch.ones(vals.shape).double()*val_var])
        # Add the outlier component
        locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).double()], dim=1)
        scales = torch.cat([scales, torch.ones(locs.shape[0], 1).double()*outlier_var], dim=1)

        # Component weights
        component_probs = torch.cat([
            torch.ones(keypoints.shape[1]).double(),
            outlier_weight], dim=0)
        component_probs = component_probs / torch.sum(component_probs)
        component_logits = torch.log(component_probs / (1. - component_probs))

        generation_dist = dist.MixtureOfDiagNormals(
            locs=torch.t(locs),
            coord_scale=torch.t(scales),
            component_logits=component_logits).expand(num_keypoints)

        observed_keypoints_and_vals = pyro.sample(
            "observed_keypoints_and_vals", generation_dist)

    def full_model():
        box = generate_single_box()
        label_origin, label_size = get_label_info_from_box_with_label(box)
        keypoints, vals = generate_keypoints_from_box_with_label(box)
        model_keypoint_observations_gmm(keypoints, vals)

    def sample_correspondences(R, scaling, t, C, model_pts, model_vals,
                               observed_pts, observed_vals,
                               spatial_variance=0.01,
                               feature_variance=0.01,
                               num_mh_iters=5,
                               outlier_prob=0.01):
        # Given a fixed affine transform B, t of a set of model points,
        # a current correspondence set C, and the point sets,
        # sample from the set of correspondences of those point sets using MH
        # and a simple flipping proposal.
        # Inspired by  "EM, MCMC, and Chain Flipping for Structure from Motion
        # with Unknown Correspondence".

        N_s = observed_pts.shape[1]
        N_m = model_pts.shape[1]
        D_spatial = model_pts.shape[0]
        assert(len(model_vals.shape) == 2)
        D_feature = model_vals.shape[0]
        # Build the pairwise distance matrix in the complete feature space
        def compute_pairwise_distances(R, scaling, t):
            # Get affine transformation of the model points
            transformed_model_pts = torch.mm(R, torch.mm(scaling, model_pts)) + t
            # Concatenate both model and observed with their vals
            m_with_vals = torch.cat([transformed_model_pts, 100.*model_vals], dim=0)
            s_with_vals = torch.cat([observed_pts, 100.*observed_vals], dim=0)
            
            # Reshape each of these into D x N_m x N_s matrix so we can take distances
            m_expanded = m_with_vals.unsqueeze(-1).permute(0, 1, 2).repeat(1, 1, N_s)
            s_expanded = s_with_vals.unsqueeze(-1).permute(0, 2, 1).repeat(1, N_m, 1)
            distances = torch.norm(m_expanded - s_expanded, dim=0)
            return distances

        pairwise_distances = compute_pairwise_distances(R, scaling, t)
        # Translate into probabilities
        pairwise_distances[0:N_m, :] = (
            torch.exp(-pairwise_distances[0:N_m, :]/(2.*spatial_variance)) /
            math.sqrt(2*np.pi*spatial_variance))
        pairwise_distances[N_m:, :] = (
            torch.exp(-pairwise_distances[N_m:, :]/(2.*feature_variance)) /
            math.sqrt(2*np.pi*feature_variance))

        # Augment with a row of the outlier probabilities
        pairwise_distances = torch.cat([pairwise_distances,
                                        torch.ones(1, N_s).double()*outlier_prob],
                                        dim=0)
        assert(pairwise_distances.shape == C.shape)

        current_score = torch.sum(C * pairwise_distances)

        for k in range(num_mh_iters):
            # Sample a new correspondence set by randomly flipping some number of correspondences.
            C_new = C.clone().detach()
            num_to_flip = dist.Binomial(probs=0.5).sample() + 1
            for flip_k in range(int(num_to_flip)):
                to_add_ind_s = torch.randint(N_s, (1, 1))
                to_add_ind_m = dist.Categorical(
                    probs=(pairwise_distances*(1. - C))[:, to_add_ind_s].squeeze()).sample()
                if to_add_ind_m != N_m:
                    C_new[to_add_ind_m, :] = 0.
                C_new[:, to_add_ind_s] = 0.
                C_new[to_add_ind_m, to_add_ind_s] = 1.
            new_score = torch.sum(C_new * pairwise_distances)

            # MH acceptance
            if torch.rand(1).item() <= new_score / current_score:
                C = C_new

        return C

    def sample_transform_given_correspondences(C, model_pts, observed_pts):
        # Given the fixed correspondences, find an optimal translation t
        # and affine transform B to align the corresponded model and observed
        # points.
        # In the case of these boxes, I'm going to use a restricted class of
        # affine transforms allowing scale but not skew: the full transform
        # is R * scaling * model_points + t = observed points
        # where scaling is a 3x3 diagonal matrix. As long as the model
        # points are 

        # Reorder into only the relevant point pairs
        keep_model_points = C.sum(dim=1) != 0
        keep_model_points[-1] = False
        C_reduced = C[keep_model_points, :]
        model_pts_reduced = model_pts[:, keep_model_points[:-1]]
        assert(model_pts_reduced.shape[1] <= observed_pts.shape[1])
        assert(all(torch.sum(C_reduced, dim=1) == 1.))
        assert(torch.sum(keep_model_points) > 0)
        corresp_inds = C_reduced.nonzero()[:, 1]
        observed_pts_reduced = observed_pts[:, corresp_inds]

        # Get R and s by alternations: see "Orthogonal, but not Orthonormal, Procrustes Problems."
        scaling = torch.eye(3).double()*0.5
        model_pts_aligned = torch.t(torch.t(model_pts_reduced) - torch.mean(model_pts_reduced, dim=1))
        observed_pts_aligned = torch.t(torch.t(observed_pts_reduced) - torch.mean(observed_pts_reduced, dim=1))

        for k in range(10):
            # Solve for rotation
            U, S, V = torch.svd(
                torch.mm(observed_pts_aligned,
                         torch.mm(torch.t(model_pts_aligned), scaling)))
            R = torch.mm(U, torch.t(V))
            # Check det and flip if necessary
            if(torch.det(R)) < 0:
                flip_eye = torch.eye(3).double()
                flip_eye[2, 2] = -1
                R = torch.mm(U, torch.mm(flip_eye, torch.t(V)))

            # Use a few steps of HMC for the remaining continuous parameters, to simulate
            # drawing from the posterior of these paramaters conditions.
            # (Parameters of interest: dimensions, label uv)

            # Closed form scaling update:
            for i in range(3):
                num = torch.sum(
                    observed_pts_aligned *
                    torch.mm(R[:, i].reshape(3, 1), model_pts_aligned[i, :].reshape(1, -1)))
                denom = torch.sum(torch.pow(model_pts_aligned[i, :], 2.))
                if torch.abs(num) > 0.01 and torch.abs(denom) > 0.01:
                    scaling[i, i] = num / denom
            scaling = torch.clamp(scaling, 0.01, 1.0)

        t = (torch.mean(observed_pts_reduced, dim=1) - torch.mean(
            torch.mm(R, torch.mm(scaling, model_pts_reduced)), dim=1)).reshape(-1, 1)

        return R, scaling, t


    def draw_pts_with_meshcat(vis, name, pts, val_channel, vals, size=0.01):
        colors = np.ones((3, pts.shape[1]))
        colors[val_channel, :] = vals[:]
        vis[name].set_object(
            g.PointCloud(position=pts, color=colors, size=size))

    def draw_corresp_with_meshcat(vis, name, pts_A, pts_B, C, size=0.01):
        pts = np.zeros((3, 2*C.shape[1]))
        assert(C.shape[0] == pts_A.shape[1] + 1)
        assert(C.shape[1] == pts_B.shape[1])
        for k in range(C.shape[1]):
            nonzero = np.nonzero(C[:, k])
            if len(nonzero) > 0 and nonzero[0] < pts_A.shape[1]:
                pts[:, 2*k] = pts_A[:, nonzero[0]]
                pts[:, 2*k+1] = pts_B[:, k]
        vis[name].set_object(
            g.LineSegments(
                g.PointsGeometry(position=pts),
                g.PointsMaterial(size=size)))

    def sample_box_from_observed_points(observed_keypoints_and_vals):
        observed_pts = torch.tensor(observed_keypoints_and_vals[:3, :]).double()
        observed_vals = torch.tensor(observed_keypoints_and_vals[3:, :]).double()

        # From the observed keypoints and values, sample correspondences and
        # a box pose.

        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")

        # Start by randomly initializing a box
        site_values = {
            "box_xyz": torch.zeros(3).double(),
            "box_quat": torch.tensor([1., 0., 0., 0.]).double(),
            "box_dimensions": torch.ones(3).double(),
            "box_label_face": torch.tensor([0]).long(),
            "box_label_uv": torch.tensor([0.5, 0.5]).double(),
            "num_observed_keypoints_minus_one": torch.tensor([observed_pts.shape[1] - 1]).long(),
            "observed_keypoints_and_vals": torch.t(torch.cat([observed_pts, observed_vals], dim=0))
        }
        def sample_model_pts_and_vals():
            box = generate_single_box()
            label_origin, label_size = get_label_info_from_box_with_label(box)
            keypoints, vals = generate_keypoints_from_box_with_label(box)
            return keypoints, vals
        model_pts, model_vals = pyro.poutine.block(
            pyro.poutine.condition(
                sample_model_pts_and_vals,
                data=site_values
            ))()

        R = torch.eye(3).double()
        scaling = torch.eye(3).double()
        t = torch.zeros(3, 1).double()
        # Correspondence matrix: one in the corresponding column of a model point and its corresponded
        # model point, plus one spare row for correspondence with "outlier"
        C = torch.eye(n=(model_pts.shape[1] + 1), m=observed_pts.shape[1])

        best_score = -1000.
        all_scores = []
        best_params = None
        all_params = []
        for k in range(200):
            print("*******ITER %03d*******" % k)

            # Regenerate model-frame model points and values given current guesses
            # of some key latents
            model_pts, model_vals = pyro.poutine.block(
                        pyro.poutine.condition(
                            sample_model_pts_and_vals,
                            data={
                                "box_xyz": torch.zeros(3).double(),
                                "box_xyz": torch.zeros(3).double(),
                                "box_quat": torch.tensor([1., 0., 0., 0.]).double(),
                                "box_dimensions": torch.ones(3).double(),
                                "box_label_face": site_values["box_label_face"],
                                "box_label_uv": site_values["box_label_uv"]
                            }
                        ))()

            C = sample_correspondences(
                R, scaling, t, C, model_pts, model_vals,
                observed_pts, observed_vals)
            if torch.sum(C[:-1, :]) > 0: # Skip transform if there are no non-outlier corresps
                R, scaling, t = sample_transform_given_correspondences(
                    C, model_pts, observed_pts)
            site_values["box_xyz"] = t.squeeze().detach()
            site_values["box_quat"] = rotMatrixToQuaternion(R).detach()
            #R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
            site_values["box_dimensions"] = torch.diag(scaling).detach()

            ## Additionally, do gradient descent on the remaining continuous params
            # TODO(gizatt) Maybe make this proper HMC?
            gd_site_names = ["box_label_uv"]
            for gd_k in range(5):
                for site_name in gd_site_names:
                    site_values[site_name].requires_grad = True
                    if site_values[site_name].grad is not None:
                        site_values[site_name].grad.data.zero_()
                # Compute score using the Pyro models
                conditioned_model = pyro.poutine.condition(
                    full_model,
                    data=site_values)
                trace = pyro.poutine.trace(conditioned_model).get_trace()
                lps = trace.log_prob_sum()
                lps.backward()
#
                #print("vals after backward: ", site_values)
                for site_name in gd_site_names:
                    site_values[site_name].data += site_values[site_name].grad * 0.0001
            site_values["box_label_uv"].data = torch.clamp(site_values["box_label_uv"].data, 0.01, 0.99)
            site_values["box_quat"].data = site_values["box_quat"].data / torch.norm(site_values["box_quat"].data)
            R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
            t = site_values["box_xyz"].reshape(-1, 1)
            scaling = torch.diag(site_values["box_dimensions"])

            # Compute score using the Pyro models
            conditioned_model = pyro.poutine.condition(
                full_model,
                data=site_values)
            trace = pyro.poutine.trace(conditioned_model).get_trace()
            lps = trace.log_prob_sum()
            lps.backward()

            #for name, site in trace.nodes.items():
            #    if site["type"] is "sample":
            #        print("Name: ", name)
            #        print("\tValue: ", site["value"])
            #        print("\tlog prob sum: ", site["log_prob_sum"])
            print("\tTotal Log prob sum: ", lps.item())

            # TODO(gizatt): Accept/reject with something like MH?
            # Problem is that my "proposal distribution" is probably not
            # symmetric, so calculating a proper MH ratio won't be as simple
            # as the likelihood ratio... for now, I'll stick to just importance
            # sampling with this relatively arbitrary proposal.

            all_scores.append(lps.detach().item())
            copied_dict = {}
            for key in site_values.keys():
                copied_dict[key] = site_values[key].clone().detach()
            all_params.append(copied_dict)
            if lps.item() > best_score:
                best_score = lps.item()
                best_params = (R, scaling, t, C)
            
            scaling = torch.diag(site_values["box_dimensions"])
            R = quaternionToRotMatrix(site_values["box_quat"].reshape(1, -1))[0, :, :]
            t = site_values["box_xyz"].reshape(-1, 1)
            model_pts_tf = torch.mm(R, torch.mm(scaling, model_pts)) + t
            draw_pts_with_meshcat(vis, "fitting/observed", 
                                  observed_pts.detach().numpy(),
                                  val_channel=0, vals=observed_vals.detach().numpy())
            draw_pts_with_meshcat(vis, "fitting/fit", 
                                  model_pts_tf.detach().numpy(),
                                  val_channel=2, vals=model_vals.detach().numpy())
            draw_corresp_with_meshcat(vis, "fitting/corresp",
                model_pts_tf.detach().numpy(), observed_pts.detach().numpy(),
                C)

        model_pts_tf = torch.mm(best_params[0], torch.mm(best_params[1], model_pts)) + best_params[2]
        draw_pts_with_meshcat(vis, "fitting/observed", 
                              observed_pts.detach().numpy(),
                              val_channel=0, vals=observed_vals.detach().numpy())
        draw_pts_with_meshcat(vis, "fitting/fit", 
                              model_pts_tf.detach().numpy(),
                              val_channel=2, vals=model_vals.detach().numpy())
        draw_corresp_with_meshcat(vis, "fitting/corresp",
            model_pts_tf.detach().numpy(), observed_pts.detach().numpy(),
            best_params[3])
        print("Best final score: ", best_score)
        print("Best final params: ", best_params)

        plt.figure()
        plt.subplot(5, 1, 1)
        plt.hist(all_scores)
        plt.xlabel("llog")
        plt.ylabel("count")

        for k in range(3):
            x = np.array([params["box_dimensions"][k].item() for params in all_params])
            plt.subplot(5, 3, 4+k)
            plt.hist(x)
            plt.xlabel("box dim %d" % k)
            plt.ylabel("count")

            plt.subplot(5, 3, 7+k)
            plt.plot(x)
            plt.ylabel("box dim %d" % k)
            plt.xlabel("epoch")

        for k in range(2):
            x = np.array([params["box_label_uv"][k].item() for params in all_params])
            plt.subplot(5, 2, 7+k)
            plt.hist(x)
            plt.xlabel("label uv %d" % k)
            plt.ylabel("count")

            plt.subplot(5, 2, 9+k)
            plt.plot(x)
            plt.ylabel("label uv %d" % k)
            plt.xlabel("epoch")

        plt.tight_layout()
        plt.show()

    from pyro.contrib.autoguide import AutoGuideList, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel
    from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
    from pyro.optim import Adam

    with open("box_observations.yaml", "r") as file:
        all_envs = yaml.load(file)

    # Pick a box at random from the envs
    observed_box_info = list(all_envs.values())[0]["obj_%04d" % 0]
    observed_keypoints = pickle.loads(observed_box_info["observed_keypoints"])
    observed_vals = pickle.loads(observed_box_info["observed_vals"])
    print("Observed box info: ", observed_box_info)
    print("Observed keypoints and vals: ", observed_keypoints, observed_vals)
    observed_keypoints_and_vals = np.vstack([observed_keypoints, observed_vals])

    print(sample_box_from_observed_points(observed_keypoints_and_vals))

if __name__ == "__main__":
    #for k in range(50):
    #    sample_and_draw_drake()
    sample_and_draw_pyro()