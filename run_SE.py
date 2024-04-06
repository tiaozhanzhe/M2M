# -*- coding: utf-8 -*-
import time
import multiprocessing
import os

architecture_list = ['mesh_HBM']

# architecture_list = ['mesh_DDR4', 'mesh_HBM', 'cmesh_DDR4', 'cmesh_HBM','ring_DDR4', 'ring_HBM',]
# network_list = ['ncf', 'VGG16', 'resnet18', 'vit', 'Unet']
# network_list = ['ncf', 'VGG16', 'resnet18', 'vit', 'Unet', 'bert_base', 'darknet19', 'GNMT', 'resnet50']
network_list = ['bert_base']

chiplet_parallel_list = ['P_stable', 'K_stable', 'PK_stable']
chiplet_num_max = '16'
chiplet_num_min = '16'

#  根据需要填写需要运行的命令
def run(network, architecture, chiplet_parallel):
    s = "python test_intralayer.py --architecture " + architecture + " --app_name " + network + "  --chiplet_num_max "+ chiplet_num_max + " --chiplet_num_min " + chiplet_num_min + " --chiplet_parallel " + chiplet_parallel +  " --PE_parallel All  --save_all_records 0  --layer_fuse_tag 0  --optimization_objective latency"
    # s = "python test_intralayer.py --architecture " + architecture + " --app_name " + network + " --alg GA  --encode_type num  --dataflow ours  --chiplet_num "+ str(chiplet_num) + " --chiplet_parallel " + chiplet_parallel +  " --PE_parallel All  --save_all_records 0  --layer_fuse_tag 0  --optimization_objective latency"

    os.system(s)

if __name__ == '__main__':
    dir = './SE'
    os.chdir(dir)
    os.system('pwd')
    start = time.time()
    
    for architecture in architecture_list:
        p_list = []
        for network in network_list:
            for chiplet_parallel in chiplet_parallel_list:
                    p_list.append(multiprocessing.Process(target=run, args=(network, architecture, chiplet_parallel)))
            # for i in range(chiplet_num_min-1, chiplet_num_max):
            #     chiplet_num = i + 1
            #     for chiplet_parallel in chiplet_parallel_list:
            #         p_list.append(multiprocessing.Process(target=run, args=(network, chiplet_parallel, chiplet_num)))

        # 启动子进程
        for p in p_list:
            p.start()

        # 等待fork的子进程终止再继续往下执行，可选填一个timeout参数
        for p in p_list:
            p.join()

    end = time.time()
    print('network_list = ', network_list)
    print('start time = ', start)
    print('end time = ', end)
    print('end - start = %f h' % ((end - start) / 3600))
