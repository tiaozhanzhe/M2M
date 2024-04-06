cp ../nn_parser_CCASM_hetero/task/single_engine_example_5_pipe/* ../dnn_task/single_engine/
./build/Garnet_standalone/gem5.opt \
	configs/example/garnet_synth_traffic.py \
	--topology=hetero_mesh_nopRouter \
	--num-cpus=126 \
	--num-dirs=126 \
	--synthetic=DNN \
	--dnn_task=single_engine \
	--link_width_bits=128 \
	--if_debug=0 \
	--network=garnet \
	--inj-vnet=2 \
	--sim-cycles=2000 \
	--injectionrate=0.02


