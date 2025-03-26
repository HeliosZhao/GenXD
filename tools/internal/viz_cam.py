import numpy as np

import open3d as o3d
import json
import os
import argparse

def camera_visualization(extrinsics, save_path, cam_c2w=True, scale=0.1, input_extrinsics=None):
    # Rts: T, 4, 4
    if cam_c2w:
        c2w = extrinsics
    else:
        c2w = np.linalg.inv(extrinsics)
    
    # camera_frames = []
    # create an empty mesh 
    
    camera_frames = o3d.geometry.TriangleMesh()
    for i in range(c2w.shape[0]):

        # Invert the extrinsic matrix to get the camera-to-world transformation
        Rt_inv = c2w[i]

        # Extract the rotation matrix and translation vector
        rotation_matrix = Rt_inv[:3, :3]
        translation_vector = Rt_inv[:3, 3]

        # Create a coordinate frame representing the camera
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0, 0, 0])
        # import ipdb; ipdb.set_trace()
        # Apply the rotation and translation to the camera frame
        camera_frame.rotate(rotation_matrix, center=[0, 0, 0])  # Rotate around the origin
        camera_frame.translate(translation_vector)
        
        # Paint the camera frame with a color gradient from blue to red
        color = [i / c2w.shape[0], 0, 1 - (i / c2w.shape[0])]
        camera_frame.paint_uniform_color(color)

        camera_frames += camera_frame    
    
    ## draw input extrinsics with another color
    if input_extrinsics is not None:
        if cam_c2w:
            c2w = input_extrinsics
        else:
            c2w = np.linalg.inv(input_extrinsics)
        for i in range(c2w.shape[0]):

            # Invert the extrinsic matrix to get the camera-to-world transformation
            Rt_inv = c2w[i]

            # Extract the rotation matrix and translation vector
            rotation_matrix = Rt_inv[:3, :3]
            translation_vector = Rt_inv[:3, 3]

            # Create a coordinate frame representing the camera
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale*2, origin=[0, 0, 0])

            # Apply the rotation and translation to the camera frame
            camera_frame.rotate(rotation_matrix, center=[0, 0, 0])
            camera_frame.translate(translation_vector)
            
            # Paint the camera frame with a color gradient from blue to red
            ## draw input extrinsics with yellow
            color = [1, 1, 0]
            camera_frame.paint_uniform_color(color)

            camera_frames += camera_frame    
            
    # merge them to one 3d file using open3d
    o3d.io.write_triangle_mesh(save_path, camera_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default="datasets/custom_cam/forward.json")
    parser.add_argument("-o", "--output_path", type=str, default=None)
    parser.add_argument("--c2w", action="store_true")
    parser.add_argument("-s", "--scale", type=float, default=0.1)
    args = parser.parse_args()
    json_path = args.data_path
    # import ipdb; ipdb.set_trace()
    # json_path = "datasets/custom_cam/forward.json"
    extrinsics = json.load(open(json_path))
    if isinstance(extrinsics, dict):
        extrinsics = extrinsics['camera']
        w2c_3x4 = np.array(extrinsics)[:,6:].reshape(-1, 3, 4)
    else:
        w2c_3x4 = np.array(extrinsics).reshape(-1, 3, 4)
    w2c = np.concatenate([w2c_3x4, np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(w2c_3x4.shape[0], axis=0)], axis=1)
    output_path = json_path.replace(".json", ".ply") if args.output_path is None else args.output_path
    camera_visualization(w2c, output_path, cam_c2w=args.c2w, scale=args.scale)
