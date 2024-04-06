
for file in /home/zhangjm/python/old/comm_pattern/trace_file/unicast/*;
do
    echo $file is running \! ;
    ./build/Garnet_standalone/gem5.opt \
	configs/example/garnet_synth_traffic.py \
	--topology=hetero_mesh_nopRouter \
	--num-cpus=252 \
	--num-dirs=252 \
	--synthetic=DNN \
	--vcs-per-vnet=1000 \
	--dnn_task=$file \
	--if_debug=0 \
	--network=garnet \
	--inj-vnet=2 \
	--sim-cycles=40000 \
	--injectionrate=0.02 \
    2>&1 | tee wxy.log
    python3 result_out_new.py $file
done