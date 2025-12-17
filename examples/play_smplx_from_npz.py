#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import glob
import argparse
import os.path as osp
from typing import List, Tuple, Optional

import numpy as np
import torch
import smplx

try:
    import open3d as o3d
except ImportError as e:
    raise ImportError("This script requires open3d. Try: pip install open3d") from e

def make_ground_plane(size: float = 4.0,
                      thickness: float = 0.01,
                      axis: str = "z",
                      height: float = 0.0):
    """
    Create a big thin box to act as a ground plane.
    axis: which axis is "up" (normal of the plane).
    height: position along that up axis where the plane surface is.
    """
    axis = axis.lower()
    if axis == "z":
        extents = (size, size, thickness)
        translate = (-size/2, -size/2, height - thickness/2)
    elif axis == "y":
        extents = (size, thickness, size)
        translate = (-size/2, height - thickness/2, -size/2)
    elif axis == "x":
        extents = (thickness, size, size)
        translate = (height - thickness/2, -size/2, -size/2)
    else:
        raise ValueError("--ground-axis must be one of: x, y, z")

    ground = o3d.geometry.TriangleMesh.create_box(*extents)
    ground.translate(translate)
    ground.compute_vertex_normals()
    ground.paint_uniform_color([0.75, 0.75, 0.75])
    return ground


def natural_key(path: str):
    """Sort paths by numeric substrings (e.g., 1,2,10 instead of 1,10,2)."""
    base = osp.basename(path)
    parts = re.split(r"(\d+)", base)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_npz_files(npz_dir: str, pattern: str = "*.npz", recursive: bool = False) -> List[str]:
    npz_dir = osp.expanduser(osp.expandvars(npz_dir))
    if recursive:
        files = glob.glob(osp.join(npz_dir, "**", pattern), recursive=True)
    else:
        files = glob.glob(osp.join(npz_dir, pattern))
    files = [f for f in files if f.lower().endswith(".npz")]
    files.sort(key=natural_key)
    return files


def _as_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim == 1:
        return x[None, ...]
    return x


def _pick_frame(x: np.ndarray, frame: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim >= 2 and x.shape[0] > 1:
        if frame < 0 or frame >= x.shape[0]:
            raise IndexError(f"Frame {frame} out of range (T={x.shape[0]})")
        return x[frame]
    if x.shape[0] == 1:
        return x[0]
    return x


def _get(npz, key: str, default=None):
    return npz[key] if key in npz.files else default


def _to_torch(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)


def build_smplx_inputs(npz, frame: int, device: str, num_betas: int, num_expr: int,
                      ignore_transl: bool = False) -> Tuple[dict, dict]:
    frame_idx = _get(npz, "frame_idx", None)
    unix_time = _get(npz, "unix_time", None)

    betas = _get(npz, "betas", None)
    body_pose = _get(npz, "body_pose", None)
    global_rot = _get(npz, "global_rot", None)
    global_trans = _get(npz, "global_trans", None)
    left_hand_pose = _get(npz, "left_hand_pose", None)
    right_hand_pose = _get(npz, "right_hand_pose", None)
    expression = _get(npz, "expression", None)

    if betas is None or body_pose is None or global_rot is None or global_trans is None:
        missing = [k for k in ["betas", "body_pose", "global_rot", "global_trans"] if k not in npz.files]
        raise KeyError(f"Missing required keys in npz: {missing}")

    betas_f = _pick_frame(_as_batch(betas), frame).astype(np.float32).reshape(-1)
    body_f  = _pick_frame(_as_batch(body_pose), frame).astype(np.float32).reshape(-1)
    grot_f  = _pick_frame(_as_batch(global_rot), frame).astype(np.float32).reshape(-1)
    gtrn_f  = _pick_frame(_as_batch(global_trans), frame).astype(np.float32).reshape(-1)

    lhp_f = np.zeros((45,), dtype=np.float32) if left_hand_pose is None else \
            _pick_frame(_as_batch(left_hand_pose), frame).astype(np.float32).reshape(-1)
    rhp_f = np.zeros((45,), dtype=np.float32) if right_hand_pose is None else \
            _pick_frame(_as_batch(right_hand_pose), frame).astype(np.float32).reshape(-1)

    expr_f = np.zeros((num_expr,), dtype=np.float32) if expression is None else \
            _pick_frame(_as_batch(expression), frame).astype(np.float32).reshape(-1)

    if betas_f.shape[0] != num_betas:
        raise ValueError(f"Expected betas dim {num_betas}, got {betas_f.shape[0]}")
    if body_f.shape[0] != 63:
        raise ValueError(f"Expected body_pose dim 63, got {body_f.shape[0]}")
    if grot_f.shape[0] != 3:
        raise ValueError(f"Expected global_rot dim 3, got {grot_f.shape[0]}")
    if gtrn_f.shape[0] != 3:
        raise ValueError(f"Expected global_trans dim 3, got {gtrn_f.shape[0]}")
    if lhp_f.shape[0] != 45 or rhp_f.shape[0] != 45:
        raise ValueError("Expected left/right hand pose dim 45 each")
    if expr_f.shape[0] != num_expr:
        raise ValueError(f"Expected expression dim {num_expr}, got {expr_f.shape[0]}")

    if ignore_transl:
        gtrn_f[:] = 0.0

    inputs = {
        "betas": _to_torch(betas_f[None, :], device),
        "body_pose": _to_torch(body_f[None, :], device),
        "global_orient": _to_torch(grot_f[None, :], device),
        "transl": _to_torch(gtrn_f[None, :], device),
        "left_hand_pose": _to_torch(lhp_f[None, :], device),
        "right_hand_pose": _to_torch(rhp_f[None, :], device),
        "expression": _to_torch(expr_f[None, :], device),
    }

    meta = {
        "frame_idx": int(frame_idx) if frame_idx is not None else None,
        "unix_time": float(unix_time) if unix_time is not None else None,
    }
    return inputs, meta


class SMPLXDirPlayer:
    def __init__(self,
                 files: List[str],
                 model,
                 device: str,
                 fps: float,
                 plot_joints: bool,
                 ignore_transl: bool,
                 seq_frame: int,
                 num_expr: int,
                 num_betas: int):
        self.files = files
        self.model = model
        self.device = device
        self.fps = fps
        self.dt = 1.0 / max(1e-6, fps)
        self.plot_joints = plot_joints
        self.ignore_transl = ignore_transl
        self.seq_frame = seq_frame
        self.num_expr = num_expr
        self.num_betas = num_betas

        self.idx = 0
        self.playing = True
        self.last_step = time.time()

        # Open3D geometries
        self.mesh = o3d.geometry.TriangleMesh()
        self.pcd = o3d.geometry.PointCloud()

        self._init_geometry()

    def _init_geometry(self):
        v, f, j, meta = self._compute(self.idx)

        self.mesh.vertices = o3d.utility.Vector3dVector(v)
        self.mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color([0.3, 0.3, 0.3])

        if self.plot_joints:
            self.pcd.points = o3d.utility.Vector3dVector(j)
            self.pcd.paint_uniform_color([0.85, 0.2, 0.2])

    def _compute(self, idx: int):
        path = self.files[idx]
        npz = np.load(path, allow_pickle=False)

        inputs, meta = build_smplx_inputs(
            npz,
            frame=self.seq_frame,
            device=self.device,
            num_betas=self.num_betas,
            num_expr=self.num_expr,
            ignore_transl=self.ignore_transl,
        )

        with torch.no_grad():
            out = self.model(**inputs, return_verts=True)

        vertices = out.vertices.detach().cpu().numpy().squeeze()
        joints = out.joints.detach().cpu().numpy().squeeze()
        faces = self.model.faces

        return vertices, faces, joints, meta

    def _apply_frame(self, vis):
        v, f, j, meta = self._compute(self.idx)

        self.mesh.vertices = o3d.utility.Vector3dVector(v)
        # faces constant; only set once, but harmless to keep as-is
        self.mesh.compute_vertex_normals()

        if self.plot_joints:
            self.pcd.points = o3d.utility.Vector3dVector(j)

        fname = osp.basename(self.files[self.idx])
        tinfo = f" unix_time={meta['unix_time']:.3f}" if meta["unix_time"] is not None else ""
        finfo = f" frame_idx={meta['frame_idx']}" if meta["frame_idx"] is not None else ""
        print(f"[{self.idx+1}/{len(self.files)}] {fname}{finfo}{tinfo}")

        vis.update_geometry(self.mesh)
        if self.plot_joints:
            vis.update_geometry(self.pcd)

    def next(self, vis):
        self.idx = (self.idx + 1) % len(self.files)
        self._apply_frame(vis)

    def prev(self, vis):
        self.idx = (self.idx - 1) % len(self.files)
        self._apply_frame(vis)

    def restart(self, vis):
        self.idx = 0
        self._apply_frame(vis)

    def toggle_play(self, vis):
        self.playing = not self.playing
        print("PLAY" if self.playing else "PAUSE")

    def tick(self, vis):
        if not self.playing:
            return
        now = time.time()
        if (now - self.last_step) >= self.dt:
            self.last_step = now
            self.next(vis)


def main():
    parser = argparse.ArgumentParser("Play SMPL-X .npz frames from a directory")
    parser.add_argument("--npz-dir", required=True, type=str, help="Directory containing .npz files")
    parser.add_argument("--pattern", default="*.npz", type=str, help="Glob pattern (default: *.npz)")
    parser.add_argument("--recursive", default=False, type=lambda s: s.lower() in ["1", "true", "yes"])

    parser.add_argument("--model-folder", required=True, type=str, help="Folder containing SMPL-X model files")
    parser.add_argument("--model-type", default="smplx",
                        choices=["smpl", "smplh", "smplx", "mano", "flame"])
    parser.add_argument("--gender", default="neutral", type=str)
    parser.add_argument("--ext", default="npz", type=str)

    parser.add_argument("--num-betas", default=10, type=int)
    parser.add_argument("--num-expression-coeffs", default=10, type=int)
    parser.add_argument("--use-face-contour", default=False, type=lambda s: s.lower() in ["1", "true", "yes"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--fps", default=15.0, type=float, help="Playback FPS")
    parser.add_argument("--plot-joints", default=False, type=lambda s: s.lower() in ["1", "true", "yes"])
    parser.add_argument("--ignore-transl", default=False, type=lambda s: s.lower() in ["1", "true", "yes"],
                        help="Zero out global translation for viewing at origin")
    parser.add_argument("--seq-frame", default=0, type=int,
                        help="If each npz stores a sequence (T, ...), pick which internal frame index to view")

    parser.add_argument("--ground", default=True, type=lambda s: s.lower() in ["1","true","yes"])
    parser.add_argument("--ground-axis", default="z", choices=["x","y","z"])
    parser.add_argument("--ground-size", default=6.0, type=float)
    parser.add_argument("--ground-thickness", default=0.01, type=float)
    parser.add_argument("--ground-auto", default=True, type=lambda s: s.lower() in ["1","true","yes"])
    parser.add_argument("--ground-height", default=0.0, type=float,
                        help="Used if --ground-auto false. Height along up axis.")
    parser.add_argument("--show-axes", default=True, type=lambda s: s.lower() in ["1","true","yes"])



    args = parser.parse_args()

    files = list_npz_files(args.npz_dir, pattern=args.pattern, recursive=args.recursive)
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in {args.npz_dir} with pattern '{args.pattern}'")

    print(f"Found {len(files)} npz files. First: {files[0]}")

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))

    # IMPORTANT: use_pca=False because your hand poses are full 45-dim axis-angle vectors.
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

    player = SMPLXDirPlayer(
        files=files,
        model=model,
        device=args.device,
        fps=args.fps,
        plot_joints=args.plot_joints,
        ignore_transl=args.ignore_transl,
        seq_frame=args.seq_frame,
        num_expr=args.num_expression_coeffs,
        num_betas=args.num_betas,
    )

    ground = None
    axes = None

    if args.ground:
        # Auto-place ground just under the first frame
        gh = args.ground_height
        if args.ground_auto:
            v0, _, _, _ = player._compute(0)
            ax = args.ground_axis.lower()
            if ax == "z":
                gh = float(np.min(v0[:, 2]))
            elif ax == "y":
                gh = float(np.min(v0[:, 1]))
            else:
                gh = float(np.min(v0[:, 0]))
            gh -= 0.02  # small margin so feet/body don't intersect the plane

        ground = make_ground_plane(
            size=args.ground_size,
            thickness=args.ground_thickness,
            axis=args.ground_axis,
            height=gh,
        )

    if args.show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="SMPL-X NPZ Player", width=1280, height=720)

    if ground is not None:
        vis.add_geometry(ground)
    if axes is not None:
        vis.add_geometry(axes)

    vis.add_geometry(player.mesh)
    if args.plot_joints:
        vis.add_geometry(player.pcd)

    # Key callbacks (Open3D uses GLFW key codes)
    KEY_SPACE = 32
    KEY_N = ord("N")
    KEY_P = ord("P")
    KEY_R = ord("R")
    KEY_Q = ord("Q")
    KEY_ESC = 256

    vis.register_key_callback(KEY_SPACE, lambda v: (player.toggle_play(v), False)[1])
    vis.register_key_callback(KEY_N,     lambda v: (player.next(v), False)[1])
    vis.register_key_callback(KEY_P,     lambda v: (player.prev(v), False)[1])
    vis.register_key_callback(KEY_R,     lambda v: (player.restart(v), False)[1])
    vis.register_key_callback(KEY_Q,     lambda v: (v.close(), False)[1])
    vis.register_key_callback(KEY_ESC,   lambda v: (v.close(), False)[1])

    print("\nControls: Space=Play/Pause | N=Next | P=Prev | R=Restart | Q/Esc=Quit\n")

    # Initial frame print
    player._apply_frame(vis)

    try:
        while vis.poll_events():
            player.tick(vis)
            vis.update_renderer()
            time.sleep(0.001)
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
