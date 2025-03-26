# Custom data generation
accelerate launch --main_process_port 1223 inference.py \
diffusion.pretrain_unet="./genxd-model" \
diffusion.output_dir="outputs/gen3d-custom" \
+evaluator.data_name="static_cam_single" \
+evaluator.data_root="data/custom" \
+evaluator.camera_info.mode="forward" +evaluator.camera_info.elevation=0. "+evaluator.camera_info.azimuth_range=[-30,30]" \
+evaluator.focal=1.0938 +evaluator.camera_info.radius=2.0


## Reconfusion format data generation
# First step: generate pose for each scene
python tools/pose_traj_generate.py -d data/reconfusion-torch/re10k -o outputs/pose_dataset --save_data_pose -sx 0.2 0.4 -0.2 -0.4 -sz 0.2 0.4 -0.2 -0.4 -n 18

# Second step: generate reconfusion data with grouped views
accelerate launch --main_process_port 1224 inference.py \
diffusion.pretrain_unet="./genxd-model" \
diffusion.output_dir="outputs/re10k-group" \
+evaluator.data_name="reconfgroup" \
+evaluator.data_root="data/reconfusion-torch/re10k" \
+evaluator.pose_dir="outputs/pose_dataset/re10k" \
+evaluator.num_context_views=3 +evaluator.n_views=3 \
+evaluator.save_target_only=True +evaluator.pad_to_square=True

# Optional: directly generate reconfusion target views
accelerate launch --main_process_port 1224 inference.py \
diffusion.pretrain_unet="./genxd-model" \
diffusion.output_dir="outputs/re10k-direct" \
+evaluator.data_name="reconfdirectgen" \
+evaluator.data_root="data/reconfusion-torch/re10k" \
+evaluator.num_context_views=3 +evaluator.n_views=3 \
+evaluator.save_target_only=True +evaluator.pad_to_square=True