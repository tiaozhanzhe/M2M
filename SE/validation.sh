
python3 validation.py \
	--architecture mesh_DDR4 \
	--app_name resnet18 \
	--chiplet_num 16 \
	--chiplet_parallel  P_K_PK \
	--PE_parallel All \
	--save_all_records 0 \
	--optimization_objective latency \
	--temporal_level 3