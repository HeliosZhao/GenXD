import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import argparse
import json
import ipdb
from scipy.interpolate import splprep, splev
from tqdm import tqdm
import os

import torch
from internal.viz_cam import camera_visualization
from internal.cam_utils import generate_ellipse_path, generate_spiral_path, generate_interpolated_path, colmap_w2c_3x4_to_nerf, nerf_c2w_3x4_to_colmap, pad_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a B-spline trajectory from a set of camera positions and orientations.')
    parser.add_argument("-d", "--data_root", type=str, help="Path to the data root directory, it can also be json file")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory", default=None)
    parser.add_argument("-n", "--num_points", type=int, help="num sample points along the B-spline", default=120)
    parser.add_argument("--num_frames", type=int, default=12)
    
    ## re10k / co3d
    parser.add_argument("-sx", "--shift_x", type=float, nargs="+", help="shift ratio on x plane for re10k", default=None)
    parser.add_argument("-sz", "--shift_z", type=float, nargs="+", help="shift ratio on y plane for re10k", default=None)
    parser.add_argument("-ms", "--multiscale", type=float, nargs="+", help="scale the trajectory for co3d", default=None)
    
    ## dtu, shift z
    parser.add_argument("--shift_zy", action="store_true", help="return center of forward facing camera")
    parser.add_argument("--z_rate", type=float, default=0., help="return center of forward facing camera")
    
    
    ## used for custom trajectory
    parser.add_argument("--return_center", action="store_true", help="return center of forward facing camera")
    parser.add_argument("--save_format", type=str, help="save format for the generated pose, can be 3x4 or 18", default="3x4")
    
    ## save all camera poses in the dataset
    parser.add_argument("--save_data_pose", action="store_true", help="save all camera poses in the dataset")
    args = parser.parse_args()
    
    data_name = os.path.basename(args.data_root)
    data_info = json.load(open(os.path.join(args.data_root, 'index.json')))
    output_dir = args.data_root if args.output_dir is None else args.output_dir
    
    for key, torch_fn in tqdm(data_info.items()):
        tqdm.write(f"Processing {key}")
        save_dir = os.path.join(output_dir, data_name, key)
        os.makedirs(save_dir, exist_ok=True)
        data = torch.load(os.path.join(args.data_root, torch_fn))[0]
        cameras = data["cameras"]
        cams = cameras.numpy() # N,18
        w2c = cams[:, 6:].reshape(-1, 3, 4) # N,3,4, numpy
        nerf_c2w = colmap_w2c_3x4_to_nerf(w2c) # N,3,4, numpy, nerf c2w
        # ipdb.set_trace()
        fix_idx = data[f"train_test_split_9"]['train_ids']
        
        scale = 0.1
        if args.save_data_pose:
            gen_poses_w2c = w2c
        else:
            if data_name == 're10k' or data_name == 'co3d':
                # re10k / co3d spline
                # ipdb.set_trace()
                gen_poses = generate_interpolated_path(
                    nerf_c2w[fix_idx], 
                    n_frames=args.num_points, # number of interpolated points per key frame 
                    spline_degree=5, 
                    smoothness=.03, 
                    rot_weight=.1,
                    shift_x=args.shift_x,
                    shift_z=args.shift_z,
                    multiscale=args.multiscale)
            elif data_name == 'llff':
                # ipdb.set_trace()
                bounds = data['poses_bounds'][:, -2:]# N,2 / 1,2
                gen_poses = generate_spiral_path(
                    nerf_c2w, 
                    bounds, 
                    n_frames=args.num_points, 
                    n_rots=1, #2, 
                    zrate=args.z_rate, #.5,
                    perc=90,
                    shift_z=args.shift_z,
                    multiscale=args.multiscale,
                    return_center=args.return_center,
                    shift_zy=args.shift_zy)
                if isinstance(gen_poses, tuple):
                    gen_poses, center = gen_poses
                    gen_poses = np.concatenate([center, gen_poses], axis=0)
                scale = 1
            elif data_name == 'dtu':
                ## llff / dtu forward facing
                gen_poses = generate_spiral_path(
                    nerf_c2w, 
                    None, 
                    n_frames=args.num_points, 
                    n_rots=1, #2, 
                    zrate=args.z_rate, #.5,
                    perc=60, # not 100, seems have more reasonable results
                    shift_z=args.shift_z,
                    multiscale=args.multiscale,
                    return_center=args.return_center,
                    shift_zy=args.shift_zy)
                scale = 1
            else:
                ## mipnerf360
                gen_poses = generate_ellipse_path(
                    nerf_c2w, 
                    n_frames=args.num_points, 
                    const_speed=True, 
                    z_variation=0., 
                    z_phase=0.,
                    shift_z=args.shift_z,
                    multiscale=args.multiscale)
                scale = 1
            # ipdb.set_trace()
            gen_poses_w2c = nerf_c2w_3x4_to_colmap(gen_poses)
            
        gen_poses_w2c = pad_poses(gen_poses_w2c)
        inputs_w2c = pad_poses(w2c[fix_idx])
        # inputs_w2c = pad_poses(w2c)
        print(f"Generated poses: {gen_poses_w2c.shape}, Inputs poses: {inputs_w2c.shape}")
        camera_visualization(gen_poses_w2c, os.path.join(save_dir, "camera_frames.ply"), cam_c2w=False, scale=scale, input_extrinsics=inputs_w2c)
        
        with open(os.path.join(save_dir, "gen_poses_w2c.json"), "w") as f:
            gen_poses_w2c = gen_poses_w2c[:,:3]
            if args.save_format == "18":
                cams = np.concatenate( [np.zeros((gen_poses_w2c.shape[0], 6)), gen_poses_w2c.reshape(-1, 12)], axis=1).tolist() # N,18
                cams = {"camera": cams}
            else:
                cams = gen_poses_w2c.tolist()
            json.dump(cams, f, indent=4)
        



        

