
python3 validation.py \
	--architecture mesh_DDR4 \
	--app_name resnet18 \
	--chiplet_num_max 16 \
	--chiplet_num_min 16 \
	--chiplet_parallel  P_K_PK \
	--PE_parallel All \
	--save_all_records 0 \
	--layer_fuse_tag 0 \
	--optimization_objective latency \
	--temporal_level 3

python3 validation.py \
	--architecture mesh_DDR4 \
	--app_name Unet \
	--chiplet_num_max 16 \
	--chiplet_num_min 16 \
	--chiplet_parallel  P_K_PK \
	--PE_parallel All \
	--save_all_records 0 \
	--layer_fuse_tag 0 \
	--optimization_objective latency \
	--temporal_level 3