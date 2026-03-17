"""
Convert ANTS tracking dataset (MOT format gt.txt) to ND2-compatible JSON format.

The ANTS dataset contains ant colony motion trajectories in indoor/outdoor scenes.
Paper: "A dataset of ant colonies' motion trajectories in indoor and outdoor scenes
        to study clustering behavior" (GigaScience, 2022)
Data source: https://data.mendeley.com/datasets/9ws98g4npw/4

Usage:
    python scripts/convert_ants_data.py --gt_dir /path/to/Ant_dataset --output_dir ./data/ants
"""

import os
import re
import json
import numpy as np
from argparse import ArgumentParser


def parse_gt_file(gt_file):
    """Parse MOT-format gt.txt file.
    
    Returns:
        data: np.ndarray of shape (N, 7), columns: frame_id, track_id, bbox_x, bbox_y, bbox_w, bbox_h, conf
    """
    rows = []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(',')
            rows.append([float(p) for p in parts])
    return np.array(rows)


def build_trajectory_matrix(data):
    """Build position matrices from MOT data.
    
    Args:
        data: (N, 7) array from parse_gt_file
    
    Returns:
        center_x: (T, V) array of x-center positions
        center_y: (T, V) array of y-center positions
        frames: sorted list of frame indices
        track_ids: sorted list of track IDs
    """
    frames = sorted(set(data[:, 0].astype(int)))
    track_ids = sorted(set(data[:, 1].astype(int)))
    frame2idx = {f: i for i, f in enumerate(frames)}
    track2idx = {t: i for i, t in enumerate(track_ids)}
    
    T, V = len(frames), len(track_ids)
    center_x = np.full((T, V), np.nan)
    center_y = np.full((T, V), np.nan)
    
    for row in data:
        f_idx = frame2idx[int(row[0])]
        t_idx = track2idx[int(row[1])]
        cx = row[2] + row[4] / 2  # bbox_x + bbox_w/2
        cy = row[3] + row[5] / 2  # bbox_y + bbox_h/2
        center_x[f_idx, t_idx] = cx
        center_y[f_idx, t_idx] = cy
    
    return center_x, center_y, frames, track_ids


def filter_complete_tracks(center_x, center_y, track_ids):
    """Keep only tracks that appear in all frames (no NaN)."""
    valid = ~np.isnan(center_x).any(axis=0)
    if valid.sum() < center_x.shape[1]:
        print(f"  Filtering: {valid.sum()}/{center_x.shape[1]} tracks are complete")
    center_x = center_x[:, valid]
    center_y = center_y[:, valid]
    track_ids = [t for t, v in zip(track_ids, valid) if v]
    return center_x, center_y, track_ids


def build_adjacency(center_x, center_y, method='knn', k=5, threshold=None):
    """Build adjacency matrix based on spatial proximity.
    
    Args:
        center_x, center_y: (T, V) position matrices
        method: 'knn' for k-nearest neighbors, 'threshold' for distance threshold, 'full' for fully connected
        k: number of nearest neighbors (for knn method)
        threshold: distance threshold (for threshold method)
    
    Returns:
        A: (V, V) adjacency matrix (binary, symmetric, no self-loops)
        G: (E, 2) edge list
    """
    V = center_x.shape[1]
    
    if method == 'full':
        A = np.ones((V, V), dtype=int) - np.eye(V, dtype=int)
    else:
        # Use average position across all frames for adjacency
        mean_x = np.nanmean(center_x, axis=0)  # (V,)
        mean_y = np.nanmean(center_y, axis=0)  # (V,)
        
        # Pairwise distance
        dx = mean_x[:, None] - mean_x[None, :]  # (V, V)
        dy = mean_y[:, None] - mean_y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        A = np.zeros((V, V), dtype=int)
        
        if method == 'knn':
            for i in range(V):
                neighbors = np.argsort(dist[i])
                # Skip self (distance 0), take k nearest
                for j in neighbors[1:k+1]:
                    A[i, j] = 1
                    A[j, i] = 1  # Symmetric
        elif method == 'threshold':
            if threshold is None:
                threshold = np.median(dist[dist > 0])
            A = ((dist > 0) & (dist < threshold)).astype(int)
    
    # Build edge list from adjacency matrix
    G = np.stack(np.nonzero(A), axis=-1)  # (E, 2)
    
    return A, G


def compute_edge_features(center_x, center_y, G):
    """Compute edge features: distance between connected nodes at each time step.
    
    Args:
        center_x, center_y: (T, V) position matrices
        G: (E, 2) edge list
    
    Returns:
        dist_edge: (T, E) distance between connected nodes
    """
    T = center_x.shape[0]
    E = G.shape[0]
    dist_edge = np.zeros((T, E))
    
    for e_idx in range(E):
        i, j = G[e_idx]
        dx = center_x[:, j] - center_x[:, i]
        dy = center_y[:, j] - center_y[:, i]
        dist_edge[:, e_idx] = np.sqrt(dx**2 + dy**2)
    
    return dist_edge


def convert_sequence(gt_file, seq_name, output_dir, fps, adjacency_method='knn', k=5,
                     subsample_step=1):
    """Convert a single sequence to ND2 format.
    
    Creates two JSON files per sequence:
    - {seq_name}_vx.json: predicting x-velocity from position
    - {seq_name}_vy.json: predicting y-velocity from position
    """
    print(f"\nConverting {seq_name}...")
    data = parse_gt_file(gt_file)
    center_x, center_y, frames, track_ids = build_trajectory_matrix(data)
    center_x, center_y, track_ids = filter_complete_tracks(center_x, center_y, track_ids)
    
    V = len(track_ids)
    T = len(frames)
    print(f"  Ants (nodes): {V}, Frames: {T}, FPS: {fps}")
    
    if V < 2:
        print(f"  Skipping: too few complete tracks")
        return
    
    # Subsample frames to reduce temporal resolution (optional)
    if subsample_step > 1:
        center_x = center_x[::subsample_step]
        center_y = center_y[::subsample_step]
        T = center_x.shape[0]
        print(f"  After subsampling (step={subsample_step}): {T} frames")
    
    # Build adjacency
    A, G = build_adjacency(center_x, center_y, method=adjacency_method, k=min(k, V-1))
    E = G.shape[0]
    print(f"  Edges: {E}, Adjacency method: {adjacency_method}")
    
    # Compute velocities (pixel displacement per frame)
    vx = (center_x[1:] - center_x[:-1]) * fps  # (T-1, V) pixels/second
    vy = (center_y[1:] - center_y[:-1]) * fps
    
    # Node state at each time step (position, normalized)
    px = center_x[:-1].copy()  # (T-1, V)
    py = center_y[:-1].copy()  # (T-1, V)
    
    # Normalize positions to [0, 1]
    px_min, px_max = px.min(), px.max()
    py_min, py_max = py.min(), py.max()
    px_norm = (px - px_min) / (px_max - px_min + 1e-8)
    py_norm = (py - py_min) / (py_max - py_min + 1e-8)
    
    # Normalize velocities
    vx_norm = vx / (np.abs(vx).max() + 1e-8)
    vy_norm = vy / (np.abs(vy).max() + 1e-8)
    
    # Edge features: inter-ant distance (normalized)
    dist_edge = compute_edge_features(center_x[:-1], center_y[:-1], G)  # (T-1, E)
    dist_norm = dist_edge / (dist_edge.max() + 1e-8)
    
    # Speed as additional node feature
    speed = np.sqrt(vx**2 + vy**2)
    speed_norm = speed / (speed.max() + 1e-8)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined dataset (predict x-velocity)
    result = dict(
        V=V,
        E=E,
        A=A.tolist(),
        G=G.tolist(),
        px=px_norm.tolist(),       # normalized x-position (T-1, V)
        py=py_norm.tolist(),       # normalized y-position (T-1, V)
        vx=vx_norm.tolist(),       # normalized x-velocity (T-1, V) - target
        vy=vy_norm.tolist(),       # normalized y-velocity (T-1, V) - target
        speed=speed_norm.tolist(), # normalized speed (T-1, V)
        M=dist_norm.tolist(),      # edge distance feature (T-1, E)
        info=dict(
            seq_name=seq_name,
            num_ants=V,
            num_frames=T,
            fps=fps,
            adjacency_method=adjacency_method,
            subsample_step=subsample_step,
            track_ids=[int(t) for t in track_ids],
        )
    )
    
    out_path = os.path.join(output_dir, f'{seq_name}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f)
    print(f"  Saved: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")


def main():
    parser = ArgumentParser(description='Convert ANTS MOT dataset to ND2 format')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Path to Ant_dataset directory containing IndoorDataset/OutdoorDataset')
    parser.add_argument('--output_dir', type=str, default='./data/ants',
                        help='Output directory for converted JSON files')
    parser.add_argument('--adjacency', type=str, default='knn',
                        choices=['knn', 'threshold', 'full'],
                        help='Adjacency matrix construction method')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of nearest neighbors (for knn adjacency)')
    parser.add_argument('--subsample', type=int, default=5,
                        help='Subsample every N frames (reduce temporal resolution)')
    parser.add_argument('--indoor_only', action='store_true',
                        help='Only convert indoor sequences (all tracks complete)')
    args = parser.parse_args()
    
    sequences = []
    for env in ['IndoorDataset', 'OutdoorDataset']:
        env_dir = os.path.join(args.gt_dir, env)
        if not os.path.exists(env_dir):
            print(f"Warning: {env_dir} not found, skipping")
            continue
        if args.indoor_only and env == 'OutdoorDataset':
            print("Skipping OutdoorDataset (--indoor_only)")
            continue
        for seq in sorted(os.listdir(env_dir)):
            gt_file = os.path.join(env_dir, seq, 'gt', 'gt.txt')
            if os.path.exists(gt_file):
                # Parse fps from environment type
                m = re.findall(r'\d+', seq)
                seq_id = int(m[0])
                fps = 25 if seq_id <= 5 else 30  # Indoor=25fps, Outdoor=30fps
                sequences.append((gt_file, seq, fps))
    
    print(f"Found {len(sequences)} sequences to convert")
    for gt_file, seq_name, fps in sequences:
        convert_sequence(gt_file, seq_name, args.output_dir, fps,
                        adjacency_method=args.adjacency, k=args.k,
                        subsample_step=args.subsample)
    
    print(f"\nDone! Converted files are in {args.output_dir}")


if __name__ == '__main__':
    main()
