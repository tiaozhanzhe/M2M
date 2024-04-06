python workload_partition.py \
	--app_name_line resnet18+VGG16+bert_base+GNMT \
	--chiplet_num_min_TH 1 \
	--chiplet_num_max_TH 16 \
	--fuse_flag 0 \
	--architecture mesh_HBM \
	--workload_par_objective iact_num \
	--workload_num_TH 4