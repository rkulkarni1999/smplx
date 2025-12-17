#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import argparse
import numpy as np
import torch
import smplx


def _as_batch(x: np.ndarray) -> np.ndarray:
    """
    Ensure array has leading batch dim.
    Accepts:
      (D,)      -> (1, D)
      (T, D)    -> (T, D)   # already batch/time
      (,)       -> (1,)     # scalar
    """
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x[None, ...]
    return x


def _pick_frame(x: np.ndarray, frame: int) -> np.ndarray:
    """
    If x is (T, ...), pick one frame -> (...).
    If x is (1, ...), return that -> (...).
    If x is scalar-batched (1,), return scalar.
    """
    x = np.asarray(x)
    if x.ndim >= 2 and x.shape[0] > 1:
        if frame < 0 or frame >= x.shape[0]:
            raise IndexError(f"--frame {frame} out of range for array with T={x.shape[0]}")
        return x[frame]
    # x is (1, ...) or (1,) -> squeeze first dim
    if x.shape[0] == 1:
        return x[0]
    return x


def _get_npz_key(npz, key, default=None):
    return npz[key] if key in npz.files else default


def _to_torch(x: np.ndarray, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


def build_smplx_inputs(npz, frame: int, device: str, num_betas: int, num_expr: int):
    """
    Returns dict of SMPL-X inputs (batched tensors).
    Your npz seems to store axis-angle for everything:
      global_rot (3,)
      global_trans (3,)
      body_pose (63,)
      left_hand_pose (45,)
      right_hand_pose (45,)
      betas (10,)
    """
    # Scalars / metadata (optional)
    frame_idx = _get_npz_key(npz, "frame_idx", None)
    unix_time = _get_npz_key(npz, "unix_time", None)

    # Core SMPL-X params
    betas = _get_npz_key(npz, "betas", None)
    body_pose = _get_npz_key(npz, "body_pose", None)
    global_rot = _get_npz_key(npz, "global_rot", None)
    global_trans = _get_npz_key(npz, "global_trans", None)
    left_hand_pose = _get_npz_key(npz, "left_hand_pose", None)
    right_hand_pose = _get_npz_key(npz, "right_hand_pose", None)

    if betas is None:
        raise KeyError("Missing 'betas' in npz")
    if body_pose is None:
        raise KeyError("Missing 'body_pose' in npz")
    if global_rot is None:
        raise KeyError("Missing 'global_rot' in npz")
    if global_trans is None:
        raise KeyError("Missing 'global_trans' in npz")

    # Make batched/time arrays
    betas_b = _as_batch(betas)
    body_b = _as_batch(body_pose)
    grot_b = _as_batch(global_rot)
    gtrans_b = _as_batch(global_trans)

    # Pick one frame if time dimension exists
    betas_f = _pick_frame(betas_b, frame)
    body_f = _pick_frame(body_b, frame)
    grot_f = _pick_frame(grot_b, frame)
    gtrans_f = _pick_frame(gtrans_b, frame)

    # Hands are optional, but your file has them
    if left_hand_pose is not None:
        lhp_b = _as_batch(left_hand_pose)
        lhp_f = _pick_frame(lhp_b, frame)
    else:
        lhp_f = np.zeros((45,), dtype=np.float32)

    if right_hand_pose is not None:
        rhp_b = _as_batch(right_hand_pose)
        rhp_f = _pick_frame(rhp_b, frame)
    else:
        rhp_f = np.zeros((45,), dtype=np.float32)

    # Expression optional
    expr = _get_npz_key(npz, "expression", None)
    if expr is not None:
        expr_b = _as_batch(expr)
        expr_f = _pick_frame(expr_b, frame)
    else:
        expr_f = np.zeros((num_expr,), dtype=np.float32)

    # Ensure expected sizes (light validation)
    betas_f = np.asarray(betas_f, dtype=np.float32).reshape(-1)
    if betas_f.shape[0] != num_betas:
        raise ValueError(f"Expected betas to have {num_betas} dims, got {betas_f.shape[0]}")

    body_f = np.asarray(body_f, dtype=np.float32).reshape(-1)
    if body_f.shape[0] != 63:
        raise ValueError(f"Expected body_pose to have 63 dims, got {body_f.shape[0]}")

    grot_f = np.asarray(grot_f, dtype=np.float32).reshape(-1)
    if grot_f.shape[0] != 3:
        raise ValueError(f"Expected global_rot to have 3 dims, got {grot_f.shape[0]}")

    gtrans_f = np.asarray(gtrans_f, dtype=np.float32).reshape(-1)
    if gtrans_f.shape[0] != 3:
        raise ValueError(f"Expected global_trans to have 3 dims, got {gtrans_f.shape[0]}")

    lhp_f = np.asarray(lhp_f, dtype=np.float32).reshape(-1)
    if lhp_f.shape[0] != 45:
        raise ValueError(f"Expected left_hand_pose to have 45 dims, got {lhp_f.shape[0]}")

    rhp_f = np.asarray(rhp_f, dtype=np.float32).reshape(-1)
    if rhp_f.shape[0] != 45:
        raise ValueError(f"Expected right_hand_pose to have 45 dims, got {rhp_f.shape[0]}")

    expr_f = np.asarray(expr_f, dtype=np.float32).reshape(-1)
    if expr_f.shape[0] != num_expr:
        raise ValueError(f"Expected expression to have {num_expr} dims, got {expr_f.shape[0]}")

    # Convert to torch with batch dim = 1
    inputs = {
        "betas": _to_torch(betas_f[None, :], device),
        "body_pose": _to_torch(body_f[None, :], device),
        "global_orient": _to_torch(grot_f[None, :], device),
        "transl": _to_torch(gtrans_f[None, :], device),
        "left_hand_pose": _to_torch(lhp_f[None, :], device),
        "right_hand_pose": _to_torch(rhp_f[None, :], device),
        "expression": _to_torch(expr_f[None, :], device),
    }

    meta = {
        "frame_idx": int(frame_idx) if frame_idx is not None else None,
        "unix_time": float(unix_time) if unix_time is not None else None,
    }
    return inputs, meta


def visualize(vertices: np.ndarray,
              faces: np.ndarray,
              joints: np.ndarray,
              plotting_module: str,
              plot_joints: bool):
    if plotting_module == "pyrender":
        import pyrender
        import trimesh

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.9]
        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.01)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)

    elif plotting_module == "open3d":
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        geometry = [mesh]
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            joints_pcl.points = o3d.utility.Vector3dVector(joints)
            joints_pcl.paint_uniform_color([0.8, 0.2, 0.2])
            geometry.append(joints_pcl)

        o3d.visualization.draw_geometries(geometry)

    elif plotting_module == "matplotlib":
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        mesh = Poly3DCollection(vertices[faces], alpha=0.2)
        mesh.set_edgecolor((0, 0, 0))
        mesh.set_facecolor((1.0, 1.0, 0.9))
        ax.add_collection3d(mesh)

        if plot_joints:
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    else:
        raise ValueError(f"Unknown plotting_module: {plotting_module}")


def main():
    parser = argparse.ArgumentParser("Visualize SMPL-X params stored in a .npz")
    parser.add_argument("--npz", required=True, type=str, help="Path to the .npz (your ground truth frame file)")
    parser.add_argument("--model-folder", required=True, type=str, help="Folder containing SMPL-X model files")
    parser.add_argument("--model-type", default="smplx", choices=["smpl", "smplh", "smplx", "mano", "flame"])
    parser.add_argument("--gender", default="neutral", type=str)
    parser.add_argument("--ext", default="npz", type=str)
    parser.add_argument("--num-betas", default=10, type=int)
    parser.add_argument("--num-expression-coeffs", default=10, type=int)
    parser.add_argument("--use-face-contour", default=False, type=lambda s: s.lower() in ["1", "true", "yes"])
    parser.add_argument("--plotting-module", default="pyrender", choices=["pyrender", "open3d", "matplotlib"])
    parser.add_argument("--plot-joints", default=False, type=lambda s: s.lower() in ["1", "true", "yes"])
    parser.add_argument("--frame", default=0, type=int, help="If the npz stores sequences, choose which frame index to view.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ignore-transl", default=False, type=lambda s: s.lower() in ["1", "true", "yes"],
                        help="If true, set transl=[0,0,0] for viewing at origin.")

    args = parser.parse_args()

    npz_path = osp.expanduser(osp.expandvars(args.npz))
    model_folder = osp.expanduser(osp.expandvars(args.model_folder))

    data = np.load(npz_path, allow_pickle=False)

    print("Loaded keys:")
    for k in data.files:
        v = data[k]
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # IMPORTANT: use_pca=False so hands accept full axis-angle (45 dims),
    # which matches your left_hand_pose/right_hand_pose arrays.
    model = smplx.create(
        model_folder,
        model_type=args.model_type,
        gender=args.gender,
        ext=args.ext,
        num_betas=args.num_betas,
        num_expression_coeffs=args.num_expression_coeffs,
        use_face_contour=args.use_face_contour,
        use_pca=False,
    ).to(args.device)

    inputs, meta = build_smplx_inputs(
        data, frame=args.frame, device=args.device,
        num_betas=args.num_betas, num_expr=args.num_expression_coeffs
    )

    if args.ignore_transl:
        inputs["transl"] = torch.zeros_like(inputs["transl"])

    print("\nMeta:")
    print(f"  frame_idx (stored): {meta['frame_idx']}")
    print(f"  unix_time (stored): {meta['unix_time']}")

    print("\nFeeding SMPL-X with:")
    for k, t in inputs.items():
        print(f"  {k}: {tuple(t.shape)} on {t.device}")

    with torch.no_grad():
        out = model(**inputs, return_verts=True)

    vertices = out.vertices.detach().cpu().numpy().squeeze()
    joints = out.joints.detach().cpu().numpy().squeeze()
    faces = model.faces

    print("\nOutput:")
    print("  vertices:", vertices.shape)
    print("  joints:", joints.shape)
    print("  faces:", faces.shape)

    visualize(vertices, faces, joints, args.plotting_module, args.plot_joints)


if __name__ == "__main__":
    main()
