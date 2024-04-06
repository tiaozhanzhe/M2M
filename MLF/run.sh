python multi_network_DSE.py \
	--architecture ours \
	--nn_list resnet18+resnet50+vit+VGG16+GNMT \
	--chiplet_num 4 \
	--Optimization_Objective latency \
	--BW_Reallocator_tag 0 \
	--layout_mapping_method balance \
	--tp_TH 4 \
	--sp_TH 4