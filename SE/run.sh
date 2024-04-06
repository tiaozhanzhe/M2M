python test_intralayer.py \
	--architecture mesh_DDR4 \
	--app_name darknet19 \
	--chiplet_num_max 16 \
	--chiplet_num_min 1 \
	--chiplet_parallel  PK_stable \
	--PE_parallel All \
	--save_all_records 0 \
	--layer_fuse_tag 0 \
	--optimization_objective latency

