import argparse
from collections import namedtuple
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
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
    vals = np.linspace(0., 1., pts.shape[1]).reshape(1, -1)
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
        label_face = np.random.choice(['p', 'n']) + \
                     np.random.choice(['x', 'y', 'z'])
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
    simulator.StepTo(1.0)
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
    vals_visible = vals[keep > 0]

    plt.figure()
    plt.imshow(depth_im, cmap='summer')
    plt.scatter(keypoints_rendered_visible[0, :],
                keypoints_rendered_visible[1, :],
                c=vals_visible, cmap="winter")
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_xaxis()

    #print("Keypoints and vals in global frame: ", keypoints, vals)
    keypoints_and_vals_single = np.vstack([keypoints, vals])[:, :9]
    np.savetxt("keypoints_obs.np", keypoints_and_vals_single[:, keep[:9] > 0])
    plt.show()


def sample_and_draw_pyro():
    import torch
    import pyro
    import pyro.infer
    import pyro.optim
    import pyro.distributions as dist
    import torch.distributions.constraints as constraints

    def generate_single_box(name="box"):
        quat_mean = pyro.param("quat_mean", torch.zeros(4).double())
        quat_scale = pyro.param(
            "quat_scale", torch.ones(4).double(),
            constraint=constraints.positive)
        quat = pyro.sample(
            name + "_quat",
            dist.Normal(quat_mean, quat_scale).to_event(1)
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
        vals = torch.linspace(0., 1., pts.shape[1]).double().reshape(1, -1)
        quat = box.pose[:4].reshape(1, -1)
        xyz = box.pose[-3:].reshape(1, -1)
        tf_mat = transParamsToHomMatrix(quat, xyz)[0]
        pts = torch.mm(tf_mat, torch.cat([pts, torch.ones(pts.shape[1]).double().reshape(1, -1)], dim=0))[:3, :]
        return pts, vals

    # This is written as a model of observed data, not a generative model --
    # i.e. I have a for-each-datapoint loop that associates each datapoint
    # with its underlying true keypoint.
    def model_keypoint_observations(keypoints, vals, observed_keypoints=None, observed_vals=None):
        # Model the assignment of each observed datapoint with a keypoint
        n_possible_keypoints = keypoints.shape[1]
        assignment_probs = pyro.param(
            "assignment_probs",
            torch.ones(n_possible_keypoints).double(),
            constraint=constraints.simplex)
        #outlier_prob = pyro.param("outlier_prob", torch.ones(1)*0.2,
        #                          constraint=constraints.unit_interval)

        n_observed_keypoints = observed_keypoints.shape[1]
        # Assume each detected point is independent, choosing from among the
        # possible keypoints / vals of this box plus being an outlier
        with pyro.plate("detections", n_observed_keypoints):
            assignment = pyro.sample(
                "assignment",
                dist.Categorical(assignment_probs),
                infer={"enumerate":"sequential"})
            #is_spurious = pyro.sample("outlier", dist.Bernoulli(outlier_prob),
            #                          infer={"enumerate":"sequential"}).type(torch.ByteTensor)

            # Ultimately model a keypoint observation as being normally distributed
            # in 3D space as well as the value space, by stacking the two
            augmented_keypoints = torch.cat([
                keypoints[:, assignment], vals[0, assignment].reshape(-1, assignment.shape[0])], dim=0)
            augmented_keypoints_obs = torch.cat([
                observed_keypoints,
                observed_vals.reshape(-1, observed_keypoints.shape[1])], dim=0)
            keypoint_emission_scale = 0.01
            #with pyro.poutine.mask(mask=is_spurious):
            #    pyro.sample("spurious_observations",
            #                dist.Normal(0., 1.).expand(augmented_keypoints_obs.shape).to_event(2),
            #                obs=augmented_keypoints_obs)
            #with pyro.poutine.mask(mask=is_spurious):
            pyro.sample("real_observations",
                        dist.Normal(augmented_keypoints_obs, keypoint_emission_scale).to_event(2),
                        obs=augmented_keypoints_obs)
        


    def conditioned_model(observed_keypoints_and_vals):
        observed_keypoints = torch.tensor(observed_keypoints_and_vals[:3, :]).double()
        observed_vals = torch.tensor(observed_keypoints_and_vals[3:, :]).double()

        box = generate_single_box()
        label_origin, label_size = get_label_info_from_box_with_label(box)
        keypoints, vals = generate_keypoints_from_box_with_label(box)
        model_keypoint_observations(keypoints, vals, observed_keypoints, observed_vals)

    def run_affine_cpd_on_model_and_observed(model_keypoints, model_vals,
                                             observed_keypoints, observed_vals,
                                             verbose=False):
        # CPD essentially operates by modeling the observed keypoints as
        # independent draws from a GMM induced by the model keypoints, where the
        # means of the GMM are constrained to be connected by an affine transform
        # of the model keypoint set, and the variances are unknown.
        # Ref https://en.wikipedia.org/wiki/Point_set_registration for its abbreviated
        # algorithm writeup.
        N_s = observed_keypoints.shape[1]
        N_m = model_keypoints.shape[1]
        D_spatial = model_keypoints.shape[0]
        w = 0.01
        assert(len(model_vals.shape) == 2)
        D_feature = model_vals.shape[0]
        def solve_affine(M, S, P, verbose=False):
            # Scene points S, model points M, association probs P
            # S is Ns x D
            # M is Nm x D
            # P is Nm x Ns
            if verbose:
                print("Solving affine with ...")
                print("\tM = ", M)
                print("\tS = ", S)
                print("\tP = ", P)
            P_rowsum = torch.sum(P, dim=1).reshape(-1, 1)
            P_colsum = torch.sum(P, dim=0).reshape(-1, 1)
            N_p = torch.sum(P_rowsum)
            mu_s = torch.mm(torch.t(S), P_colsum) / N_p  # D x 1
            mu_m = torch.mm(torch.t(M), P_rowsum) / N_p  # D x 1
            S_hat = torch.t(torch.t(S) - torch.sum(mu_s)) # Take off mean translations
            M_hat = torch.t(torch.t(M) - torch.sum(mu_m))
            # Now compute B and t using those terms
            lhs = torch.mm(torch.t(S_hat), torch.mm(torch.t(P), M_hat))
            rhs = torch.mm(torch.t(M_hat), torch.mm(torch.diag(P_rowsum.squeeze()), M_hat))
            rhs = torch.inverse(rhs)
            B = torch.mm(lhs, rhs)
            t = mu_s - torch.mm(B, mu_m)
            # And compute variance using those terms.
            lhs = torch.trace(torch.mm(torch.t(S_hat),
                                       torch.mm(torch.diag(P_colsum.squeeze()), S_hat)))
            rhs = torch.trace(torch.mm(torch.t(S_hat),
                                       torch.mm(torch.t(P),
                                                torch.mm(M_hat, torch.t(B)))))
            variance =  100.*(lhs - rhs)/(N_p * S.shape[1])
            if verbose:
                print("\tResulting B = ", B)
                print("\tResulting t = ", t)
                print("\tResulting var = ", variance)
            return B, t, variance

        def compute_pairwise_distances(B, t):
            # Get affine transformation of the model points
            transformed_model_keypoints = torch.mm(B, model_keypoints) + t
            # Concatenate both model and observed with their vals
            m_with_vals = torch.cat([transformed_model_keypoints, 100.*model_vals], dim=0)
            s_with_vals = torch.cat([observed_keypoints, 100.*observed_vals], dim=0)
            
            # Reshape each of these into D x N_m x N_s matrix so we can take distances
            m_expanded = m_with_vals.unsqueeze(-1).permute(0, 1, 2).repeat(1, 1, N_s)
            s_expanded = s_with_vals.unsqueeze(-1).permute(0, 2, 1).repeat(1, N_m, 1)
            distances = torch.norm(m_expanded - s_expanded, dim=0)
            return distances

        # Initial affine TF guess
        random_quat = torch.randn(1, 4)
        random_quat = random_quat / torch.norm(random_quat)
        B = torch.mm(torch.eye(3)*(1. + torch.randn(1)*0.1), quaternionToRotMatrix(random_quat)[0, :, :])
        B = B.double()
        t = torch.randn(3, 1).double()
        # Initialize variance as average distance between all pairs of points
        distances = compute_pairwise_distances(B, t)
        print("Initial distances: ", distances)
        variance = torch.sum(distances)
        variance = variance / (N_s * N_m * (D_feature + D_spatial))
        print("Init variance: ", variance.item())
        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        for k in range(10):
            # Bound variance
            variance = torch.clamp(variance, min=1E-4, max=None)
            # Compute the association probabilities under the current GMM parameters guesses
            # Slight deviation from vanilla CPD: the distance between a pair of points
            # is going to be a combination of the Euclidean distance between the points' features
            # and the distance between the affine-transformed model point with the scene point.
            distances = compute_pairwise_distances(B, t)
            # And now just form the complete probability matrix by using the normalization
            matchwise_probs = torch.exp(-distances/(2.*variance))
            print("Matchwise probs: ", matchwise_probs)
            c = torch.pow((2 * np.pi * variance), (D_feature + D_spatial)/2.) + (w / (1. - w)) * (N_m / N_s)
            col_normalizers = torch.sum(matchwise_probs, dim=0) + c
            P = matchwise_probs / col_normalizers.reshape(1, -1).repeat(N_m, 1)
            if torch.matrix_rank(P) < 3:
                if verbose:
                    print("Failed to find a fit.")
                break
            if verbose:
                print("P before: ", P)
                print("B, t, var before: ", B, t, variance)
            B, t, variance = solve_affine(torch.t(model_keypoints), torch.t(observed_keypoints), P)
            if verbose:
                print("B, t, var after: ", B, t, variance)

            print("Var at iter %d: %f" % (k, variance.item()))
            model_pts_tf = torch.mm(B, model_keypoints) + t
            draw_pts_with_meshcat(vis, "fitting/fit", 
                                  model_pts_tf.detach().numpy(),
                                  val_channel=2, vals=model_vals.detach().numpy())
            time.sleep(1.)

        return B, t, P, variance

    def draw_pts_with_meshcat(vis, name, pts, val_channel, vals, size=0.01):
        colors = np.ones((3, pts.shape[1]))
        colors[val_channel, :] = vals[:]
        vis[name].set_object(
            g.PointCloud(position=pts, color=colors, size=size))

    def sample_box_from_observed_points(observed_keypoints_and_vals):
        observed_keypoints = torch.tensor(observed_keypoints_and_vals[:3, :]).double()
        observed_vals = torch.tensor(observed_keypoints_and_vals[3:, :]).double()

        # From the observed keypoints and values, sample correspondences and
        # a box pose.

        # Start by randomly initializing a box
        sampled_box = pyro.poutine.block(generate_single_box)()

        # Greedily correspond observed keypoints to the keypoints of that
        # sampled box
        model_pts, model_vals = pyro.poutine.block(generate_keypoints_from_box_with_label)(sampled_box)

        #observed_keypoints = model_pts
        #observed_vals = model_vals
        B, t, P, variance = run_affine_cpd_on_model_and_observed(
            model_pts.detach(), model_vals.detach(),
            observed_keypoints.detach(), observed_vals.detach())
        model_pts_tf = torch.mm(B, model_pts) + t

        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        
        draw_pts_with_meshcat(vis, "fitting/observed", 
                              observed_keypoints.detach().numpy(),
                              val_channel=0, vals=observed_vals.detach().numpy())
        draw_pts_with_meshcat(vis, "fitting/fit", 
                              model_pts_tf.detach().numpy(),
                              val_channel=2, vals=model_vals.detach().numpy())

        print("Aligned:")
        print("\tvariance: ", variance)
        print("\tB: ", B)
        print("\tt: ", t)
        print("\tP: ", P)



    if 1:
        from pyro.contrib.autoguide import AutoGuideList, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel
        from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
        from pyro.optim import Adam

        observed_keypoints_and_vals = np.loadtxt("keypoints_obs.np")
        print(sample_box_from_observed_points(observed_keypoints_and_vals))
        sys.exit(0)

        guide = AutoGuideList(conditioned_model)
        discrete_sites = ["box_label_face", "assignment"]
        guide.add(AutoDelta(
            pyro.poutine.block(conditioned_model, hide=discrete_sites)))
        #guide.add(AutoDiscreteParallel(
        #    pyro.poutine.block(conditioned_model, expose=discrete_sites)))
        optim = Adam({'lr': 0.1})
        svi = SVI(conditioned_model, guide, optim, TraceEnum_ELBO())

        for step in range(100):
            loss = svi.step(observed_keypoints_and_vals)
            print("%03d: %07.07f" % (step, loss))

        for param_name in pyro.get_param_store().keys():
            print("%s: " % param_name, pyro.param(param_name))
        
        
    if 0:
        from pyro.infer.mcmc import HMC
        from pyro.infer.mcmc.api import MCMC
        nuts_kernel = HMC(
            conditioned_model,
            max_plate_nesting=1,
            target_accept_prob=0.8)
        mcmc = MCMC(nuts_kernel,
                    num_samples=10,
                    warmup_steps=25,
                    num_chains=1)
        mcmc.run()
        mcmc.summary(prob=0.5)
        print(mcmc.get_samples())

if __name__ == "__main__":
    #sample_and_draw_drake()
    sample_and_draw_pyro()