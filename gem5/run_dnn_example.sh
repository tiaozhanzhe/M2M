./build/Garnet_standalone/gem5.opt \
	configs/example/garnet_synth_traffic.py \
	--topology=hetero_mesh_nopRouter \
	--num-cpus=252 \
	--num-dirs=252 \
	--vcs-per-vnet=20 \
	--synthetic=DNN \
	--dnn_task=/home/wangxy/chiplet/comm_pattern/trace_file/unicast/c16_resnet50_layer6 \
	--if_debug=0 \
	--network=garnet \
	--inj-vnet=2 \
	--sim-cycles=40000 \
	--injectionrate=0.02 \
	 2>&1 | tee wxy.log
