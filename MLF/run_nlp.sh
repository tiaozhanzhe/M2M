python multi_network_DSE.py \
	--architecture ring_DDR4 \
	--nn_list GNMT+bert_base+ncf \
	--chiplet_num 8 \
	--Optimization_Objective latency \
	--BW_Reallocator_tag 1 \
	--layout_mapping_method balance \
	--tp_TH 4 \
	--sp_TH 3