python multi_network_DSE.py \
	--architecture mesh_HBM \
	--nn_list VGG16+resnet18+GNMT+bert_base \
	--chiplet_num 16 \
	--Optimization_Objective latency \
	--BW_Reallocator_tag 1 \
	--layout_mapping_method balance \
	--tp_TH 1 \
	--sp_TH 4