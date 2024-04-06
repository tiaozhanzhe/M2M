python workload_partition.py \
	--app_name_line GNMT+bert_base+ncf \
	--chiplet_num_min_TH 1 \
	--chiplet_num_max_TH 8 \
	--fuse_flag 0 \
	--architecture ring_DDR4 \
	--workload_par_objective iact_num \
	--workload_num_TH 4