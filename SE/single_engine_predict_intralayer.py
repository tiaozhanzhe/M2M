import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from numpy.core.fromnumeric import mean
import re
np.set_printoptions(threshold=sys.maxsize)
from pathlib import Path
from GaEncode import *
from config import *
from multicast_method import *

wgt_tag =  (int(1001))
act_tag =  (int(1002))
out_tag =  (int(1003))

debug = 0
# 建立DNN模型
#print ("task = ",task)
#print (DNN1.layer_list[0].i_H)


#### a NoP+NoC example #########
# 硬件信息
# memory_param = {"OL1":,"OL2":,"AL1":,"AL2":,"WL1":,"WL2":}
# 卷积配置 从basicParam_noc_nop中import进来

def calPSumAllReduce(output_num, chiplet_num, PC3 ):
    output_flit_num = int(output_num / neu_per_flit_psum_nop)
    delay = (output_flit_num / chiplet_num) * 2 * (PC3-1)
    d2d_energy = (output_num * psum_width / chiplet_num) * 2 * (PC3-1) * chiplet_num * DIE2DIE_energy_ratio
    dram_energy = (output_num * psum_width / chiplet_num) * PC3 * 2 * chiplet_num * DRAM_energy_ratio
    energy_list = [d2d_energy, dram_energy, d2d_energy+dram_energy]
    return delay, energy_list


def calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, network_param, HW_param, memory_param, NoC_param, if_multicast, architecture, i_act_SRAM_enough=0, o_act_DRAM=0, weight_SRAM_enough=0, fuse_par_num=1, fuse_tag = "initial", io_die_tag = 1):
    route_table = NoC_param["route_table"]
    bw_scales = NoC_param["bw_scales"]
    F = NoC_param["F"]
    link_energy_ratio = NoC_param["energy_ratio"]
    link_energy = F.copy()
    # 映射方案 (目前只实现了K维度有并行度)
    # Ga Encode
    #for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GaGetChild()
    CoreNum = HW_param["PE"][0] * HW_param["PE"][1]
    PE_lenth = HW_param["PE"][1]
    PE_height = HW_param["PE"][0]
    ChipNum = HW_param["Chiplet"][0] * HW_param["Chiplet"][1]
    Chip_lenth = HW_param["maxChiplet"][1]
    Chip_height = HW_param["maxChiplet"][0]
    maxChipNum = Chip_lenth * Chip_lenth
    port_average = (4 + (Chip_lenth-2+Chip_height-2)*2*2 + (Chip_height-2)*(Chip_lenth-2)*4) / maxChipNum
    OL1 = memory_param["OL1"]
    OL2 = memory_param["OL2"]
    AL1 = memory_param["AL1"]
    AL2 = memory_param["AL2"]
    WL1 = memory_param["WL1"]
    WL2 = memory_param["WL2"]
    data_flow = for_list[0]
    ol1_ratio = for_list[1]
    al1_ratio = for_list[2]
    wl1_ratio = for_list[3]
    all_param = for_list[4]
    out_final = for_list[5]
    if_act_share_PE = for_list[6]
    if_wgt_share_PE = for_list[7]
    if_act_share_Chiplet = for_list[8]
    if_wgt_share_Chiplet = for_list[9]

    # mapping parameter
    P1,P2,P3 = partition_list["P"][0],partition_list["P"][1],partition_list["P"][2]
    Q1,Q2,Q3 = partition_list["Q"][0],partition_list["Q"][1],partition_list["Q"][2]
    C1,C2,C3 = partition_list["C"][0],partition_list["C"][1],partition_list["C"][2]
    K1,K2,K3 = partition_list["K"][0],partition_list["K"][1],partition_list["K"][2]
    PP2,PQ2,PC2,PK2 = parallel_dim_list[0][0],parallel_dim_list[0][1],parallel_dim_list[0][2],parallel_dim_list[0][3]
    PP3,PQ3,PC3,PK3 = parallel_dim_list[1][0],parallel_dim_list[1][1],parallel_dim_list[1][2],parallel_dim_list[1][3]
    PK0 = HW_param["intra_PE"]["K"]
    PC0 = HW_param["intra_PE"]["C"]

    # network parameter
    P = network_param["P"]
    Q = network_param["Q"]
    K = network_param["K"]
    C = network_param["C"]
    R = network_param["R"]
    S = network_param["S"]
    stride = network_param["stride"]

    # memory node id
    ol2_node = PE_height * (PE_lenth + 1) + A_W_offset['o']

    al2_node = ol2_node + PE_lenth + 1
    wl2_node = ol2_node + (PE_lenth + 1) * 2
    dram_node  = 0


    runtimeP = PP3*P3*PP2*P2*P1
    runtimeQ = PQ3*Q3*PQ2*Q2*Q1
    runtimeK = PK3*K3*PK2*K2*K1*PK0
    runtimeC = PC3*C3*PC2*C2*C1*PC0
    runtimeR = R # R S不拆分,在PE level的时序for参数里
    runtimeS = S
    runtimeCoreNum = PK2*PQ2*PP2*PC2
    runtimeChipNum = PP3*PQ3*PK3*PC3

    assert(runtimeP>=P);assert(runtimeQ>=Q);assert(runtimeK>=K);assert(runtimeC>=C)
    assert(runtimeCoreNum <= CoreNum);assert(runtimeChipNum <= ChipNum)

    energy_MAC = P*Q*K*C*R*S * MAC_energy_ratio
    compuation_num = runtimeP*runtimeQ*runtimeK*runtimeC*runtimeR*runtimeS
    compuation_cycles = compuation_num/runtimeCoreNum/runtimeChipNum/PC0/PK0
    #print ("compuation_num=",compuation_num)
    #print ("compuation_cycles=",compuation_cycles)

	# io die : ddr bandwidth

    if 'DDR4' in architecture:
        ddr_bandwidth	= 225 # Gbs (3600MHz 225Gbps)
    elif 'HBM' in architecture:
        ddr_bandwidth	= 512 * 2 # Gbps
    else:
        raise NotImplementedError

    ddr_bandwidth_unit = ddr_bandwidth / 4
    minChipNumIODie = runtimeChipNum % 4
    if minChipNumIODie == 0:
        minChipNumIODie = 4
    ddr_bandwidth_dict_no_reuse = {"input":ddr_bandwidth_unit, "weight":ddr_bandwidth_unit, "output":ddr_bandwidth_unit}
    ddr_bandwidth_io_die_method_dict = {}
    if io_die_tag == 1:
        PX3_list = [PP3*PQ3,PK3,PC3]
        PX3_reuse_dim = ["weight","input","output"]
        for id in range(len(PX3_list)):
            if PX3_list[id] >= minChipNumIODie:
                reuse_tag = PX3_reuse_dim[id]
                ddr_bandwidth_dict = copy.deepcopy(ddr_bandwidth_dict_no_reuse)
                ddr_bandwidth_dict[reuse_tag] *= minChipNumIODie
                ddr_bandwidth_io_die_method_dict[reuse_tag] = copy.deepcopy(ddr_bandwidth_dict)
            else:
                pass
        if len(ddr_bandwidth_io_die_method_dict) == 0 and minChipNumIODie == runtimeChipNum:
            ddr_bandwidth_dict = copy.deepcopy(ddr_bandwidth_dict_no_reuse)
            ddr_bandwidth_dict["input"] *= PX3_list[1]
            ddr_bandwidth_dict["weight"] *= PX3_list[0]
            ddr_bandwidth_dict["output"] *= PX3_list[2]
            ddr_bandwidth_io_die_method_dict["unique"] = copy.deepcopy(ddr_bandwidth_dict)
    else:
        ddr_bandwidth_dict_no_reuse = {"input":ddr_bandwidth, "weight":ddr_bandwidth, "output":ddr_bandwidth}
    if len(ddr_bandwidth_io_die_method_dict) == 0:
        ddr_bandwidth_io_die_method_dict["unique"] = copy.deepcopy(ddr_bandwidth_dict_no_reuse)
			
    #if io_die_tag == 1:
    #    if ChipNum == 16:
            #if PP3 == ChipNum:
            #    ddr_bandwidth_io_die_input /= 4
            #if PK3 == ChipNum or PK3 == 4:
            #    ddr_bandwidth_io_die_weight /= 4
            #mesh
    #        if PC3 == 4 or PC3 == 16:
    #            ddr_bandwidth_io_die_input /= 4
    #            ddr_bandwidth_io_die_weight /= 4
    #        elif PK3 == ChipNum:
    #            ddr_bandwidth_io_die_weight /= 4
    #    elif ChipNum == 4:
    #        if PP3 == 4:
    #            ddr_bandwidth_io_die_input /= 4
    #        elif PP3 == 2:
    #            ddr_bandwidth_io_die_input /= 2
    #            ddr_bandwidth_io_die_weight /= 2
    #        elif PP3 == 1:
    #            ddr_bandwidth_io_die_weight /= 4
    #    ddr_bandwidth_io_die_output = ddr_bandwidth / 4

    # storage size
    AL1_mem = AL1*8*1024/act_wgt_width/2 # /2是因为ping-pong
    OL1_mem = OL1*8*1024/psum_width/2 
    WL1_mem = WL1*8*1024/act_wgt_width/2
    AL2_mem = AL2*8*1024/act_wgt_width/2
    OL2_mem = OL2*8*1024/psum_width/2
    WL2_mem = WL2*8*1024/act_wgt_width/2 
    A_PE_mem = PC0
    W_PE_mem = PC0*PK0	

    OL1_need = {}; AL1_need = {}; WL1_need = {}; L1_need = {}
    OL2_need = {}; AL2_need = {}; WL2_need = {}; L2_need = {}

    cal_cycles = {}
    if_out_final = {}


    ol1_need = PK0; al1_need_CKpart= PC0; wl1_need = PK0*PC0; cal =1
    al1_need_Qpart = 1; al1_need_Ppart = 1; al1_need_Rpart = 1; al1_need_Spart = 1
    # ------------------ 计算6个buffer存储需求&每级for循环循环次数 ------------------

    for id in range(len(data_flow)):
        param = data_flow[id]
        ol1_need = ol1_need * ol1_ratio[id] # 单位:neuron

        # al1 need calculation
        if "C" == param[0]:
            al1_need_CKpart = al1_need_CKpart * all_param[id]
        elif "Q" == param[0]:
            al1_need_Qpart = al1_need_Qpart * all_param[id]
        elif "P" == param[0]:
            al1_need_Ppart = al1_need_Ppart * all_param[id]
        elif "R" == param[0]:
            al1_need_Rpart = al1_need_Rpart * all_param[id]
        elif "S" == param[0]:
            al1_need_Spart = al1_need_Spart * all_param[id]

        al1_need_Q_final = al1_need_Qpart * stride + al1_need_Spart - stride
        al1_need_P_final = al1_need_Ppart * stride + al1_need_Rpart - stride
        al1_need = al1_need_CKpart * al1_need_Q_final * al1_need_P_final

        
        wl1_need = wl1_need * wl1_ratio[id]
        cal = cal * all_param[id]
        cal_cycles[param] = cal
        OL1_need[param] = ol1_need
        AL1_need[param] = al1_need
        WL1_need[param] = wl1_need
        L1_need[param] = wl1_need + al1_need + ol1_need
        if_out_final[param] = out_final[id]
        # L2
        OL2_need[param] = ol1_need * PK2 * PQ2 * PP2
        al2_need_Qpart = al1_need_Qpart * PQ2 
        al2_need_Ppart = al1_need_Ppart * PP2        

        al2_need_Q_final = al2_need_Qpart * stride + al1_need_Spart - stride
        al2_need_P_final = al2_need_Ppart * stride + al1_need_Rpart - stride
        al2_need = al1_need_CKpart * al2_need_Q_final * al2_need_P_final * PC2
        
        AL2_need[param] = al2_need #这里有点问题
        WL2_need[param] = wl1_need * PK2  * PC2

    repeat = 1
    repeat_num = {}
        
    for id in range(len(data_flow)):
        real_id = len(data_flow) - id -1
        param = data_flow[real_id] 
        repeat = repeat * all_param[real_id]
        repeat_num[param] = repeat

    # ------------------ 决定存储临界点 ------------------

    def find_cp(the_data_flow,storage_need,storage_size):
        for id in range(len(the_data_flow)):
            param = the_data_flow[id]
            if storage_need[param] > storage_size: 
                the_cp = param
                the_cp_id = id
                break
            the_cp = "top"
            the_cp_id = id
        utilization_ratio = storage_need[the_data_flow[the_cp_id-1]] / storage_size
        return the_cp,the_cp_id,utilization_ratio

    ol1_cp,ol1_cp_id,ol1_utilization_ratio = find_cp(data_flow,OL1_need,OL1_mem)
    al1_cp,al1_cp_id,al1_utilization_ratio = find_cp(data_flow,AL1_need,AL1_mem)
    wl1_cp,wl1_cp_id,wl1_utilization_ratio = find_cp(data_flow,WL1_need,WL1_mem)
    ol2_cp,ol2_cp_id,ol2_utilization_ratio = find_cp(data_flow,OL2_need,OL2_mem)
    al2_cp,al2_cp_id,al2_utilization_ratio = find_cp(data_flow,AL2_need,AL2_mem)
    wl2_cp,wl2_cp_id,wl2_utilization_ratio = find_cp(data_flow,WL2_need,WL2_mem)
    ape_cp,ape_cp_id,ape_utilization_ratio = find_cp(data_flow,AL1_need,A_PE_mem)
    wpe_cp,wpe_cp_id,wpe_utilization_ratio = find_cp(data_flow,WL1_need,W_PE_mem)

    if debug == 1:
        print("Debug in find_cp:")
        print("---OL1_mem:{} OL1_need:{}".format(OL1_mem, OL1_need))
        print("---ol1_cp:{} ol1_cp_id:{}".format(ol1_cp, ol1_cp_id))
        print("---AL1_mem:{} AL1_need:{}".format(AL1_mem, AL1_need))
        print("---al1_cp:{} al1_cp_id:{}".format(al1_cp, al1_cp_id))
        print("---WL1_mem:{} WL1_need:{}".format(WL1_mem, WL1_need))
        print("---wl1_cp:{} wl1_cp_id:{}".format(wl1_cp, wl1_cp_id))

    # ------------------ 构建mem cal core 位置和属性等 ------------------
    # 从wxy import进来

    act_core_dict = act_wgt_dict["act_core"][0]["recv"]
    wgt_core_dict = act_wgt_dict["wgt_core"][0]["recv"]
    act_chip_dict = act_wgt_dict["act_chiplet"]["recv"]
    wgt_chip_dict = act_wgt_dict["wgt_chiplet"]["recv"]
    out_core_dict = out_dict["rd_core"][0]["recv"]
    out_chip_dict = out_dict["rd_chip"]["recv"]

    # 依据信息构建 mem_node_list 和 cc_node_list 
    mem_node_list = [ol2_node,al2_node,wl2_node,dram_node]
    cc_node_list = []
    for item in wgt_core_dict:
        core_id_list = wgt_core_dict[item]
        for core_id in core_id_list:
            if core_id not in cc_node_list:
                cc_node_list.append(core_id)

    # ------------------ 性能预测：计算整层所有计算和通信数据的数目 ------------------
    # REG <-> L1 用于统计通信总量 & prediction
    pe_neu_num_rd_wgt = 0 # 单位 neuron数目 
    pe_neu_num_rd_act = 0

    cur = data_flow[ape_cp_id]; inner = data_flow[ape_cp_id-1]  
    if ape_cp == "top":
        pe_neu_num_rd_act += AL1_need[inner] * 1
    else:
        pe_neu_num_rd_act += AL1_need[inner] * repeat_num[cur]

    cur = data_flow[wpe_cp_id]; inner = data_flow[wpe_cp_id-1]  
    pe_neu_num_rd_wgt += WL1_need[inner] * repeat_num[cur]

    pe_neu_num_rd_wgt = pe_neu_num_rd_wgt * CoreNum * ChipNum # 考虑到片上有CoreNum * ChipNum个PE
    pe_neu_num_rd_act = pe_neu_num_rd_act * CoreNum * ChipNum # 考虑到片上有CoreNum * ChipNum个PE
    energy_rd_wgt_L1 = pe_neu_num_rd_wgt * SRAM_energy(WL1) * act_wgt_width 
    energy_rd_act_L1 = pe_neu_num_rd_act * SRAM_energy(AL1) * act_wgt_width  

    # L1 用于统计通信总量 & prediction
    core_pkt_num_wr_opt = 0; core_neu_num_wr_opt = 0  # 单位分别是 packet | neuron数目 
    core_pkt_num_rd_opt = 0; core_neu_num_rd_opt = 0
    core_pkt_num_rd_wgt = 0; core_neu_num_rd_wgt = 0
    core_pkt_num_rd_act = 0; core_neu_num_rd_act = 0

    # L1 用于生成task file的变量
    core_rd_out_data_num = 0
    core_out_data_num = 0 
    core_act_data_num = 0
    core_wgt_data_num = 0

    cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
    if (if_out_final[cur]!=1): 
        #print("CORE: read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
        core_pkt_num_rd_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_psum)) * repeat_num[cur]
        core_neu_num_rd_opt += OL1_need[inner] * repeat_num[cur]
        core_rd_out_data_num += OL1_need[inner]
    else:
        core_pkt_num_rd_opt += 0
        core_neu_num_rd_opt += 0
        core_rd_out_data_num += 0
    #print("CORE: write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
    if (if_out_final[cur]!=1):
        core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
    else:
        core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    core_out_data_num += OL1_need[inner] # 用于生成仿真指令
    core_neu_num_wr_opt += OL1_need[inner] * repeat_num[cur]
        
    cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
    #print("CORE: read act mem ",AL1_need[inner],"repeat ",repeat_num[cur])
    core_pkt_num_rd_act +=  int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
    core_act_data_num += AL1_need[inner] # 用于生成仿真指令
    if al1_cp == "top":
        core_neu_num_rd_act += AL1_need[inner] * 1
    else:
        core_neu_num_rd_act += AL1_need[inner] * repeat_num[cur]

    cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
    #print("CORE: read wgt mem ",WL1_need[inner],"repeat ",repeat_num[cur]) 
    core_pkt_num_rd_wgt += int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    core_wgt_data_num += WL1_need[inner] # 用于生成仿真指令
    core_neu_num_rd_wgt += WL1_need[inner] * repeat_num[cur]

    # 考虑上并行度带来的数据复用机会 (多播)
    if if_multicast == 1:
        core_neu_num_wr_opt = core_neu_num_wr_opt * CoreNum * ChipNum  # 没有机会复用
        core_neu_num_rd_opt = core_neu_num_rd_opt * CoreNum * ChipNum 
        core_neu_num_rd_wgt = core_neu_num_rd_wgt * CoreNum * ChipNum /PP2 / PQ2
        core_neu_num_rd_act = core_neu_num_rd_act * CoreNum * ChipNum /PK2 
    elif if_multicast == 0:
        core_neu_num_wr_opt = core_neu_num_wr_opt * CoreNum * ChipNum  # 没有机会复用
        core_neu_num_rd_opt = core_neu_num_rd_opt * CoreNum * ChipNum 
        core_neu_num_rd_wgt = core_neu_num_rd_wgt * CoreNum * ChipNum  
        core_neu_num_rd_act = core_neu_num_rd_act * CoreNum * ChipNum  

    energy_l2 = SRAM_energy(OL2)

    energy_l2_w = energy_l2


    if (if_out_final[data_flow[ol1_cp_id]]!=1):
        energy_wr_opt_L2 = core_neu_num_wr_opt * energy_l2 * psum_width 
    else: 
        energy_wr_opt_L2 = core_neu_num_wr_opt * energy_l2 * act_wgt_width 
    energy_rd_opt_L2 = core_neu_num_rd_opt * energy_l2 * psum_width
    energy_rd_wgt_L2 = core_neu_num_rd_wgt * energy_l2_w * act_wgt_width
    energy_rd_act_L2 = core_neu_num_rd_act * energy_l2 * act_wgt_width

    # L2 用于统计通信总量 & prediction
    chip_pkt_num_wr_opt = 0; chip_neu_num_wr_opt = 0
    chip_pkt_num_rd_opt = 0; chip_neu_num_rd_opt = 0
    chip_pkt_num_rd_wgt = 0; chip_neu_num_rd_wgt = 0
    chip_pkt_num_rd_act = 0; chip_neu_num_rd_act = 0

    # L2 用于生成task file的变量
    chip_rd_out_data_num = 0
    chip_out_data_num = 0 
    chip_act_data_num = 0
    chip_wgt_data_num = 0

    cur = data_flow[ol2_cp_id]; inner = data_flow[ol2_cp_id-1]  
    if (if_out_final[cur]!=1): 
        #print("Chip: read opt mem ", OL2_need[inner],"repeat ",repeat_num[cur]) 
        chip_pkt_num_rd_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) * repeat_num[cur]
        chip_rd_out_data_num += OL2_need[inner]
        chip_neu_num_rd_opt += OL2_need[inner] * repeat_num[cur]
    else:
        chip_pkt_num_rd_opt += 0
        chip_rd_out_data_num += 0
        chip_neu_num_rd_opt += 0
    #print("Chip: write opt mem ", OL2_need[inner],"repeat ",repeat_num[cur])

	# -- update in 22.7.20 : 一旦片上放得下所有的输出结果，就不再将输出输出到DRAM
    if (if_out_final[cur]!=1): 
        chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
    elif ol2_cp == "top" and o_act_DRAM == 0:
        chip_pkt_num_wr_opt += 0
    else:
        chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    if ol2_cp == "top":
        chip_out_data_num += 0 # 用于生成仿真指令
        chip_neu_num_wr_opt += 0
    else:
        chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
        chip_neu_num_wr_opt += OL2_need[inner] * repeat_num[cur]

    #if (if_out_final[cur]!=1): 
    #    chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
    #else:
    #    chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    #chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
    #chip_neu_num_wr_opt += OL2_need[inner] * repeat_num[cur]
        
    cur = data_flow[al2_cp_id]; inner = data_flow[al2_cp_id-1]  
    #print("Chip: read act mem ",AL2_need[inner],"repeat ",repeat_num[cur])
    assert(fuse_tag == "tailFuse" or fuse_tag == "initial" or fuse_tag == "headFuse")
    chip_pkt_num_rd_act = {"DRAM":0, "chiplet":0}
    if al2_cp == "top":
        if i_act_SRAM_enough:
            chip_pkt_num_rd_act["chiplet"] = int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * 1
        else:
            chip_pkt_num_rd_act["DRAM"] = int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * 1
        #if fuse_tag == "tailFuse":
        #    chip_pkt_num_rd_act += 0
        #else:
        #    chip_pkt_num_rd_act += int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * 1
    else:
        if i_act_SRAM_enough:
            chip_pkt_num_rd_act["chiplet"] = int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * repeat_num[cur] * 1
        else:
            chip_pkt_num_rd_act["DRAM"] = int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * repeat_num[cur] * 1
        #chip_pkt_num_rd_act += int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) * repeat_num[cur]
    
    #chip_pkt_num_rd_act +=  int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
    chip_act_data_num += AL2_need[inner] # 用于生成仿真指令
    chip_neu_num_rd_act = {"DRAM":0, "chiplet":0}
    if al2_cp == "top":
        if i_act_SRAM_enough:
            chip_neu_num_rd_act["chiplet"] = AL2_need[inner] * 1
        else:
            chip_neu_num_rd_act["DRAM"] = AL2_need[inner] * 1
        #if fuse_tag == "tailLayer":
        #    chip_neu_num_rd_act += 0
        #else:
        #    chip_neu_num_rd_act += AL2_need[inner] * 1
    else:
        if i_act_SRAM_enough:
            chip_neu_num_rd_act["chiplet"] = AL2_need[inner] * repeat_num[cur]
        else:
            chip_neu_num_rd_act["DRAM"] = AL2_need[inner] * repeat_num[cur]      
        #chip_neu_num_rd_act += AL2_need[inner] * repeat_num[cur]

    cur = data_flow[wl2_cp_id]; inner = data_flow[wl2_cp_id-1]  
    #print("Chip: read wgt mem ",WL2_need[inner],"repeat ",repeat_num[cur]) 
    chip_pkt_num_rd_wgt = {"DRAM":0, "chiplet":0}

    if weight_SRAM_enough:
        chip_pkt_num_rd_wgt["DRAM"] = math.ceil(WL2_need["top"]/flit_per_pkt/neu_per_flit_act_wgt)
        chip_pkt_num_rd_wgt["chiplet"] = int(math.ceil(WL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur] - chip_pkt_num_rd_wgt["DRAM"]
        chip_pkt_num_rd_wgt["DRAM"] = math.ceil(chip_pkt_num_rd_wgt["DRAM"] / fuse_par_num)
    else:
        chip_pkt_num_rd_wgt["DRAM"] = int(math.ceil(WL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]

    chip_wgt_data_num += WL2_need[inner] # 用于生成仿真指令
    chip_neu_num_rd_wgt = {"DRAM":0, "chiplet":0}

    if weight_SRAM_enough:
        chip_neu_num_rd_wgt["DRAM"] = WL2_need["top"]
        chip_neu_num_rd_wgt["chiplet"] = WL2_need[inner] *repeat_num[cur] - chip_pkt_num_rd_wgt["DRAM"]
        chip_neu_num_rd_wgt["DRAM"] = math.ceil(chip_neu_num_rd_wgt["DRAM"] / fuse_par_num)
    else:
        chip_neu_num_rd_wgt["DRAM"] = WL2_need[inner] *repeat_num[cur]
    #chip_neu_num_rd_wgt += WL2_need[inner] * repeat_num[cur]

    # 考虑上并行度带来的数据复用机会
    if if_multicast == 1:
        chip_neu_num_wr_opt = chip_neu_num_wr_opt * ChipNum  # 没有机会复用
        chip_neu_num_rd_opt = chip_neu_num_rd_opt * ChipNum 
        chip_neu_num_rd_wgt["DRAM"] = chip_neu_num_rd_wgt["DRAM"] * ChipNum /PP3 / PQ3
        chip_neu_num_rd_wgt["chiplet"] = chip_neu_num_rd_wgt["chiplet"] * ChipNum /PP3 / PQ3
        chip_neu_num_rd_act["DRAM"] = chip_neu_num_rd_act["DRAM"] * ChipNum /PK3
        chip_neu_num_rd_act["chiplet"] = chip_neu_num_rd_act["chiplet"] * ChipNum /PK3  
    elif if_multicast == 0:
        chip_neu_num_wr_opt = chip_neu_num_wr_opt * ChipNum  # 没有机会复用
        chip_neu_num_rd_opt = chip_neu_num_rd_opt * ChipNum 

        chip_neu_num_rd_wgt["DRAM"] = chip_neu_num_rd_wgt["DRAM"] * ChipNum
        chip_neu_num_rd_wgt["chiplet"] = chip_neu_num_rd_wgt["chiplet"] * ChipNum
        chip_neu_num_rd_act["DRAM"] = chip_neu_num_rd_act["DRAM"] * ChipNum 
        chip_neu_num_rd_act["chiplet"] = chip_neu_num_rd_act["chiplet"]

    if (if_out_final[cur]!=1): 
        energy_wr_opt_dram = chip_neu_num_wr_opt * DRAM_energy_ratio * psum_width 
    else:
        energy_wr_opt_dram = chip_neu_num_wr_opt * DRAM_energy_ratio * act_wgt_width 
    energy_rd_opt_dram = chip_neu_num_rd_opt * DRAM_energy_ratio * psum_width
    energy_rd_wgt_dram = chip_neu_num_rd_wgt["DRAM"] * DRAM_energy_ratio * act_wgt_width
    energy_rd_wgt_L2 += chip_neu_num_rd_wgt["chiplet"] * energy_l2 * act_wgt_width
    energy_rd_act_L2 += chip_neu_num_rd_act["chiplet"] * energy_l2 * act_wgt_width
    energy_rd_act_dram = chip_neu_num_rd_act["DRAM"] * DRAM_energy_ratio * act_wgt_width
	# -- update in 22.7.20 : 当片上放得下所有的activation的时候，默认activation来自于其他chiplet的L2
    #if i_act_SRAM_enough == 1:
    #    energy_rd_act_L2 += chip_neu_num_rd_act * energy_l2 * act_wgt_width
    #    energy_rd_act_dram = 0
    #else:
    #    energy_rd_act_dram = chip_neu_num_rd_act * DRAM_energy_ratio * act_wgt_width

    F_cur=F.copy()

    # 对core构建通信需求
    # 用到的信息: core_pkt_num_wr_opt; core_pkt_num_rd_opt; core_pkt_num_rd_wgt; core_pkt_num_rd_act
    bw_needed = (core_pkt_num_rd_act) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle 
    for i, item in enumerate(act_core_dict):
        dst_list = act_core_dict[item]

        if A_W_offset['o'] != 0:
            if i < len(act_core_dict) / 2:
                al2_node_tmp = al2_node
            else:
                al2_node_tmp = al2_node	+ (PE_lenth + 1) * 2
        else:
            al2_node_tmp = al2_node

        if if_multicast == 0:
            for dst in dst_list:
                for link in route_table[(al2_node_tmp + 1000, dst + 1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        elif if_multicast == 1:
            link_set = simple_multicast(al2_node_tmp + 1000, [dst + 1000 for dst in dst_list], route_table) 
            for link in link_set:
                F_cur[link] += ( bw_needed / bw_scales[link] )

    bw_needed = (core_pkt_num_rd_wgt) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
    for item in wgt_core_dict:
        dst_list = wgt_core_dict[item]
        if if_multicast == 0:
            for dst in dst_list:
                for link in route_table[(wl2_node + 1000, dst + 1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        elif if_multicast == 1:
            link_set = simple_multicast(wl2_node + 1000, [dst + 1000 for dst in dst_list], route_table) 
            for link in link_set:
                F_cur[link] += ( bw_needed / bw_scales[link] )


    bw_needed = (core_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
    
    for item in out_core_dict:
        dst_list = out_core_dict[item]
        if if_multicast == 0:
            for dst in dst_list:
                for link in route_table[(ol2_node + 1000, dst + 1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        elif if_multicast == 1:
            link_set = simple_multicast(ol2_node + 1000, [dst + 1000 for dst in dst_list], route_table) 
            for link in link_set:
                F_cur[link] += ( bw_needed / bw_scales[link] )


    bw_needed = (core_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
    for item in out_core_dict:
        dst_list = out_core_dict[item]
        for dst in dst_list:					#写output不存在多播可能
            for link in route_table[(dst + 1000, ol2_node+1000)]:
                F_cur[link] += ( bw_needed / bw_scales[link] )

    # 对chip构建通信需求
    dram_to_L2_F_cur = L2_to_DRAM_F_cur = 0
    bw_needed_io_die = {}
    bw_needed_rd_nop = 0
    # 用到的信息: chip_pkt_num_wr_opt; chip_pkt_num_rd_opt; chip_pkt_num_rd_wgt; chip_pkt_num_rd_act
    bw_needed = (chip_pkt_num_rd_act["DRAM"]) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle 
    bw_needed_io_die["input"] = bw_needed
    bw_needed_rd_nop += (chip_pkt_num_rd_act["chiplet"]) * flit_per_pkt  / compuation_cycles
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_input/noc_bandwidth)
    
    bw_needed = (chip_pkt_num_rd_wgt["DRAM"]) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
    bw_needed_io_die["weight"] = bw_needed
    bw_needed_rd_nop += (chip_pkt_num_rd_wgt["chiplet"]) * flit_per_pkt  / compuation_cycles
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_weight/noc_bandwidth)

    bw_needed = (chip_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
    bw_needed_io_die["output"] = bw_needed
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_output/noc_bandwidth)

    bw_needed = (chip_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
    bw_needed_io_die["output"] += bw_needed
    #L2_to_DRAM_F_cur += bw_needed / (ddr_bandwidth_io_die_output/noc_bandwidth)
    L2_to_DRAM_F_cur = 0

    bw_needed_io_die_order = sorted(bw_needed_io_die.items(), key = lambda x: x[1])
    data_type_order = [bw_needed_io_die_order[2][0], bw_needed_io_die_order[1][0], bw_needed_io_die_order[0][0], "unique"]
    for data_type in data_type_order:
        if data_type in ddr_bandwidth_io_die_method_dict:
            ddr_bandwidth_io_die_input = ddr_bandwidth_io_die_method_dict[data_type]["input"]
            ddr_bandwidth_io_die_weight = ddr_bandwidth_io_die_method_dict[data_type]["weight"]
            ddr_bandwidth_io_die_output = ddr_bandwidth_io_die_method_dict[data_type]["output"]

    dram_to_L2_F_cur += bw_needed_io_die["input"] / (ddr_bandwidth_io_die_input/noc_bandwidth)
    dram_to_L2_F_cur += bw_needed_io_die["weight"] / (ddr_bandwidth_io_die_weight/noc_bandwidth)
    dram_to_L2_F_cur += bw_needed_io_die["output"] / (ddr_bandwidth_io_die_output/noc_bandwidth)

    nop_F_cur = bw_needed_rd_nop / (port_average * nop_bandwidth / noc_bandwidth)
    if architecture == 'mesh_DDR4' or architecture == 'mesh_HBM' :
        nop_F_cur *= 2
    elif architecture == 'cmesh_DDR4' or architecture == 'cmesh_HBM':
        pass
    elif architecture == 'ring_DDR4' or architecture == 'ring_HBM':
        nop_F_cur *= 3

    F_cur[(ol2_node, ol2_node + 1000)] = 0
    F_cur[(ol2_node + 1000, ol2_node)] = 0
    F_cur[(al2_node + 1000, al2_node)] = 0
    F_cur[(wl2_node + 1000, wl2_node)] = 0
    degrade_ratio_dict = {"NoC":max(F_cur.values()), "L2_to_DRAM":L2_to_DRAM_F_cur, "DRAM_to_L2":dram_to_L2_F_cur, "NoP":nop_F_cur}
    degrade_ratio = max ( max(F_cur.values()), L2_to_DRAM_F_cur, dram_to_L2_F_cur, nop_F_cur)
    if (degrade_ratio < 1):
            degrade_ratio = 1
    # print ("F_cur",F_cur)
    # print ("degrade_ratio",degrade_ratio)
    runtime_calNum = runtimeP*runtimeQ*runtimeR*runtimeS*runtimeC*runtimeK
    runtime_list = [runtimeP, runtimeQ, runtimeC, runtimeK, runtimeChipNum, runtimeCoreNum,runtime_calNum]
    cp_list = [ol1_cp_id, al1_cp_id, wl1_cp_id, ol2_cp_id, al2_cp_id, wl2_cp_id]
    utilization_ratio_list = [ol1_utilization_ratio,al1_utilization_ratio,wl1_utilization_ratio, \
                              ol2_utilization_ratio,al2_utilization_ratio,wl2_utilization_ratio]
    energy_L1_list = [energy_rd_wgt_L1, energy_rd_act_L1]
    energy_dram_list = [energy_wr_opt_dram, energy_rd_opt_dram, energy_rd_wgt_dram, energy_rd_act_dram]
    energy_L2_list = [energy_wr_opt_L2, energy_rd_opt_L2, energy_rd_wgt_L2, energy_rd_act_L2]
    energy_die2die = 0;	energy_core2core = 0
    assert(DIE2DIE_energy_ratio!=NOC_energy_ratio)
    for item in link_energy:
        if link_energy_ratio[item] == DIE2DIE_energy_ratio:
            energy_die2die += link_energy[item]
        elif link_energy_ratio[item] == NOC_energy_ratio:
            energy_core2core += link_energy[item]
        elif link_energy_ratio[item] == DIE2DIE_energy_ratio + DRAM_energy_ratio:
            energy_die2die += link_energy[item]
            energy_dram_list[2] += link_energy[item]
        else:
            print ("FATAL: link's energy ratio is incorrect!")
            sys.exit()
    if PC3 > 1:
        output_num = runtimeP * runtimeQ * runtimeK
        chiplet_num = runtimeChipNum
        delay_psum, energy_psum_list = calPSumAllReduce(output_num, chiplet_num, PC3)
    else:
        delay_psum = 0
        energy_psum_list = [0,0,0]

    worstlinks = []
    for item in F_cur:
        if F_cur[item] == degrade_ratio: 
            worstlinks.append(item)
        if dram_to_L2_F_cur == degrade_ratio:
            worstlinks.append("dram2L2")
        if L2_to_DRAM_F_cur == degrade_ratio:
            worstlinks.append("L2toDRAM")
        if nop_F_cur == degrade_ratio:
            worstlinks.append("NoP")
	
    flit_needed = {}
    flit_needed["input_DRAM"] = (chip_pkt_num_rd_act["DRAM"]) * flit_per_pkt
    flit_needed["weight_DRAM"] = (chip_pkt_num_rd_wgt["DRAM"]) * flit_per_pkt
    flit_needed["input_L2"] = (chip_pkt_num_rd_act["chiplet"]) * flit_per_pkt
    flit_needed["weight_L2"] = (chip_pkt_num_rd_wgt["chiplet"]) * flit_per_pkt
    flit_needed["output_rd"] = (chip_pkt_num_rd_opt) * flit_per_pkt
    flit_needed["output_wr"] = (chip_pkt_num_wr_opt) * flit_per_pkt
    flit_needed["chiplet_parallel"] = [PK3,PQ3,PP3,PC3]

    return(degrade_ratio*compuation_cycles, degrade_ratio, degrade_ratio_dict, flit_needed, compuation_cycles,runtime_list,cp_list,utilization_ratio_list, \
        energy_dram_list, energy_L2_list,energy_L1_list, energy_die2die, energy_MAC, energy_psum_list, delay_psum, worstlinks)

# end 性能测评

def createTaskFile(dir_name, for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, network_param, HW_param, memory_param, NoC_param, all_sim_node_num, if_multicast, io_die_tag=1):
    route_table = NoC_param["route_table"]
    bw_scales = NoC_param["bw_scales"]
    F = NoC_param["F"]
    link_energy_ratio = NoC_param["energy_ratio"]

    CoreNum = HW_param["PE"][0] * HW_param["PE"][1]
    PE_lenth = HW_param["PE"][1]
    PE_height = HW_param["PE"][0]
    ChipNum = HW_param["Chiplet"][0] * HW_param["Chiplet"][1]

    OL1 = memory_param["OL1"]
    OL2 = memory_param["OL2"]
    AL1 = memory_param["AL1"]
    AL2 = memory_param["AL2"]
    WL1 = memory_param["WL1"]
    WL2 = memory_param["WL2"]

    data_flow = for_list[0]
    ol1_ratio = for_list[1]
    al1_ratio = for_list[2]
    wl1_ratio = for_list[3]
    all_param = for_list[4]
    out_final = for_list[5]
    if_act_share_PE = for_list[6]
    if_wgt_share_PE = for_list[7]
    if_act_share_Chiplet = for_list[8]
    if_wgt_share_Chiplet = for_list[9]

    # mapping parameter
    P1,P2,P3 = partition_list["P"][0],partition_list["P"][1],partition_list["P"][2]
    Q1,Q2,Q3 = partition_list["Q"][0],partition_list["Q"][1],partition_list["Q"][2]
    C1,C2,C3 = partition_list["C"][0],partition_list["C"][1],partition_list["C"][2]
    K1,K2,K3 = partition_list["K"][0],partition_list["K"][1],partition_list["K"][2]
    PP2,PQ2,PC2,PK2 = parallel_dim_list[0][0],parallel_dim_list[0][1],parallel_dim_list[0][2],parallel_dim_list[0][3]
    PP3,PQ3,PC3,PK3 = parallel_dim_list[1][0],parallel_dim_list[1][1],parallel_dim_list[1][2],parallel_dim_list[1][3]
    PK0 = HW_param["intra_PE"]["K"]
    PC0 = HW_param["intra_PE"]["C"]

    # network parameter
    P = network_param["P"]
    Q = network_param["Q"]
    K = network_param["K"]
    C = network_param["C"]
    R = network_param["R"]
    S = network_param["S"]
    stride = network_param["stride"]
    # memory node id
    # memory node id
    ol2_node = PE_height * (PE_lenth + 1) + A_W_offset['o']
    if PE_height == 2:
        al2_node = ol2_node + PE_lenth + 1
        wl2_node = ol2_node + PE_lenth + 1
    else:
        assert(PE_height > 1)
        al2_node = ol2_node + PE_lenth + 1
        wl2_node = ol2_node + (PE_lenth + 1) * 2
    dram_node  = 0


    runtimeP = PP3*P3*PP2*P2*P1
    runtimeQ = PQ3*Q3*PQ2*Q2*Q1
    runtimeK = PK3*K3*PK2*K2*K1*PK0
    runtimeC = PC3*C3*PC2*C2*C1*PC0
    runtimeR = R
    runtimeS = S
    runtimeCoreNum = PK2*PQ2*PP2*PC2
    runtimeChipNum = PP3*PQ3*PK3*PC3
    print("runtimeCoreNum ", runtimeCoreNum)
    print("runtimeChipNum ", runtimeChipNum)

    assert(runtimeP>=P);assert(runtimeQ>=Q);assert(runtimeK>=K);assert(runtimeC>=C)
    assert(runtimeCoreNum <= CoreNum);assert(runtimeChipNum <= ChipNum)

    compuation_num = runtimeP*runtimeQ*runtimeK*runtimeC*runtimeR*runtimeS
    compuation_cycles = compuation_num/runtimeCoreNum/runtimeChipNum/PC0/PK0
    #print ("compuation_num=",compuation_num)
    #print ("compuation_cycles=",compuation_cycles)

    #-------------------------------------------------#
    #----------------- IO-DIE DDR-BW -----------------#
    #-------------------------------------------------#
    ddr_bandwidth_unit = ddr_bandwidth / 4
    minChipNumIODie = runtimeChipNum % 4
    if minChipNumIODie == 0:
        minChipNumIODie = 4
    ddr_bandwidth_dict_no_reuse = {"input":ddr_bandwidth_unit, "weight":ddr_bandwidth_unit, "output":ddr_bandwidth_unit}
    ddr_bandwidth_io_die_method_dict = {}
    if io_die_tag == 1:
        PX3_list = [PP3*PQ3,PK3,PC3]
        PX3_reuse_dim = ["weight","input","output"]
        for id in range(len(PX3_list)):
            if PX3_list[id] >= minChipNumIODie:
                reuse_tag = PX3_reuse_dim[id]
                ddr_bandwidth_dict = copy.deepcopy(ddr_bandwidth_dict_no_reuse)
                ddr_bandwidth_dict[reuse_tag] *= minChipNumIODie
                ddr_bandwidth_io_die_method_dict[reuse_tag] = copy.deepcopy(ddr_bandwidth_dict)
            else:
                pass
        if len(ddr_bandwidth_io_die_method_dict) == 0 and minChipNumIODie == runtimeChipNum:
            ddr_bandwidth_dict = copy.deepcopy(ddr_bandwidth_dict_no_reuse)
            ddr_bandwidth_dict["input"] *= PX3_list[1]
            ddr_bandwidth_dict["weight"] *= PX3_list[0]
            ddr_bandwidth_dict["output"] *= PX3_list[2]
            ddr_bandwidth_io_die_method_dict["unique"] = copy.deepcopy(ddr_bandwidth_dict)
    else:
        ddr_bandwidth_dict_no_reuse = {"input":ddr_bandwidth, "weight":ddr_bandwidth, "output":ddr_bandwidth}
    if len(ddr_bandwidth_io_die_method_dict) == 0:
        ddr_bandwidth_io_die_method_dict["unique"] = copy.deepcopy(ddr_bandwidth_dict_no_reuse)

    # storage size
    AL1_mem = AL1*8*1024/act_wgt_width/2 # /2是因为ping-pong
    OL1_mem = OL1*8*1024/psum_width/2 
    WL1_mem = WL1*8*1024/act_wgt_width/2
    AL2_mem = AL2*8*1024/act_wgt_width/2
    OL2_mem = OL2*8*1024/psum_width/2
    WL2_mem = WL2*8*1024/act_wgt_width/2 

    OL1_need = {}; AL1_need = {}; WL1_need = {}; L1_need = {}
    OL2_need = {}; AL2_need = {}; WL2_need = {}; L2_need = {}

    cal_cycles = {}
    if_out_final = {}

    ol1_need = PK0; al1_need_CKpart= PC0; wl1_need = PK0*PC0; cal =1
    al1_need_Qpart = 1; al1_need_Ppart = 1; al1_need_Rpart = 1; al1_need_Spart = 1

    # ------------------ 计算6个buffer存储需求&每级for循环循环次数 ------------------

    for id in range(len(data_flow)):
        param = data_flow[id]
        ol1_need = ol1_need * ol1_ratio[id] # 单位:neuron

        # al1 need calculation
        if "C" == param[0]:
            al1_need_CKpart = al1_need_CKpart * all_param[id]
        elif "Q" == param[0]:
            al1_need_Qpart = al1_need_Qpart * all_param[id]
        elif "P" == param[0]:
            al1_need_Ppart = al1_need_Ppart * all_param[id]
        elif "R" == param[0]:
            al1_need_Rpart = al1_need_Rpart * all_param[id]
        elif "S" == param[0]:
            al1_need_Spart = al1_need_Spart * all_param[id]

        al1_need_Q_final = al1_need_Qpart * stride + al1_need_Spart - stride
        al1_need_P_final = al1_need_Ppart * stride + al1_need_Rpart - stride
        al1_need = al1_need_CKpart * al1_need_Q_final * al1_need_P_final

        
        wl1_need = wl1_need * wl1_ratio[id]
        cal = cal * all_param[id]
        cal_cycles[param] = cal
        OL1_need[param] = ol1_need
        AL1_need[param] = al1_need
        WL1_need[param] = wl1_need
        L1_need[param] = wl1_need + al1_need + ol1_need
        if_out_final[param] = out_final[id]
        # L2
        OL2_need[param] = ol1_need * PK2 * PQ2 * PP2
        al2_need_Qpart = al1_need_Qpart * PQ2 
        al2_need_Ppart = al1_need_Ppart * PP2
        al2_need_Q_final = al2_need_Qpart * stride + al2_need_Qpart - stride
        al2_need_P_final = al2_need_Ppart * stride + al2_need_Ppart - stride
        al2_need = (al1_need_CKpart * al2_need_Q_final * al2_need_P_final) * PC2
        
        AL2_need[param] = al2_need
        WL2_need[param] = wl1_need * PK2 * PC2

    repeat = 1
    repeat_num = {}
        
    for id in range(len(data_flow)):
        real_id = len(data_flow) - id -1
        param = data_flow[real_id] 
        repeat = repeat * all_param[real_id]
        repeat_num[param] = repeat

    # ------------------ 决定存储临界点 ------------------

    def find_cp(the_data_flow,storage_need,storage_size):
        for id in range(len(the_data_flow)):
            param = the_data_flow[id]
            if storage_need[param] > storage_size: 
                the_cp = param
                the_cp_id = id
                break
            the_cp = "top"
            the_cp_id = id
        return the_cp,the_cp_id

    ol1_cp,ol1_cp_id = find_cp(data_flow,OL1_need,OL1_mem)
    al1_cp,al1_cp_id = find_cp(data_flow,AL1_need,AL1_mem)
    wl1_cp,wl1_cp_id = find_cp(data_flow,WL1_need,WL1_mem)
    ol2_cp,ol2_cp_id = find_cp(data_flow,OL2_need,OL2_mem)
    al2_cp,al2_cp_id = find_cp(data_flow,AL2_need,AL2_mem)
    wl2_cp,wl2_cp_id = find_cp(data_flow,WL2_need,WL2_mem)

    #print ("OL1_need",OL1_need); print ("OL2_need",OL2_need)
    #print ("AL1_need",AL1_need); print ("AL2_need",AL2_need)
    #print ("WL1_need",WL1_need); print ("WL2_need",WL2_need)

    #print ("repeat_num",repeat_num)
    #print ("cal_cycles",cal_cycles)
    #print ("ol1_cp=",ol1_cp,"al1_cp=",al1_cp,"wl1_cp=",wl1_cp)
    #print ("ol2_cp=",ol2_cp,"al2_cp=",al2_cp,"wl2_cp=",wl2_cp)

    # ------------------ 构建mem cal core 位置和属性等 ------------------
    # 从wxy import进来

    act_core_dict = act_wgt_dict["act_core"][0]["recv"]
    wgt_core_dict = act_wgt_dict["wgt_core"][0]["recv"]
    act_chip_dict = act_wgt_dict["act_chiplet"]["recv"]
    wgt_chip_dict = act_wgt_dict["wgt_chiplet"]["recv"]
    out_core_dict = out_dict["rd_core"][0]["recv"]
    out_chip_dict = out_dict["rd_chip"]["recv"]

    # 依据信息构建 mem_node_list 和 cc_node_list 
    mem_node_list = [ol2_node,al2_node,wl2_node,dram_node]
    cc_node_list = []
    for item in wgt_core_dict:
        core_id_list = wgt_core_dict[item]
        for core_id in core_id_list:
            if core_id not in cc_node_list:
                cc_node_list.append(core_id)

    # ------------------ 性能预测：计算整层所有计算和通信数据的数目 ------------------
    # L1 用于统计通信总量 & prediction
    core_pkt_num_wr_opt = 0
    core_pkt_num_rd_opt = 0
    core_pkt_num_rd_wgt = 0
    core_pkt_num_rd_act = 0

    # L1 用于生成task file的变量
    core_rd_out_data_num = 0
    core_out_data_num = 0 
    core_act_data_num = 0
    core_wgt_data_num = 0

    cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
    if (if_out_final[cur]!=1): 
        #print("CORE: read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
        core_pkt_num_rd_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_psum)) * repeat_num[cur]
        core_rd_out_data_num += OL1_need[inner]
    else:
        core_pkt_num_rd_opt += 0
        core_rd_out_data_num += 0
   # print("CORE: write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
    if (if_out_final[cur]!=1):
        core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
    else:
        core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    core_out_data_num += OL1_need[inner] # 用于生成仿真指令
        
    cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
    #print("CORE: read act mem ",AL1_need[inner],"repeat ",repeat_num[cur])
    core_pkt_num_rd_act +=  int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
    core_act_data_num += AL1_need[inner] # 用于生成仿真指令

    cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
    #print("CORE: read wgt mem ",WL1_need[inner],"repeat ",repeat_num[cur]) 
    core_pkt_num_rd_wgt += int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    core_wgt_data_num += WL1_need[inner] # 用于生成仿真指令

    # L2 用于统计通信总量 & prediction
    chip_pkt_num_wr_opt = 0
    chip_pkt_num_rd_opt = 0
    chip_pkt_num_rd_wgt = 0
    chip_pkt_num_rd_act = 0

    # L2 用于生成task file的变量
    chip_rd_out_data_num = 0
    chip_out_data_num = 0 
    chip_act_data_num = 0
    chip_wgt_data_num = 0

    cur = data_flow[ol2_cp_id]; inner = data_flow[ol2_cp_id-1]  
    if (if_out_final[cur]!=1): 
        #print("Chip: read opt mem ", OL2_need[inner],"repeat ",repeat_num[cur]) 
        chip_pkt_num_rd_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) * repeat_num[cur]
        chip_rd_out_data_num += OL2_need[inner]
    else:
        chip_pkt_num_rd_opt += 0
        chip_rd_out_data_num += 0
    #print("Chip: write opt mem ", OL2_need[inner],"repeat ",repeat_num[cur])
    if (if_out_final[cur]!=1): 
        chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
    else:
        chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
        
    cur = data_flow[al2_cp_id]; inner = data_flow[al2_cp_id-1]  
    #print("Chip: read act mem ",AL2_need[inner],"repeat ",repeat_num[cur])
    chip_pkt_num_rd_act +=  int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
    chip_act_data_num += AL2_need[inner] # 用于生成仿真指令

    cur = data_flow[wl2_cp_id]; inner = data_flow[wl2_cp_id-1]  
    #print("Chip: read wgt mem ",WL2_need[inner],"repeat ",repeat_num[cur]) 
    chip_pkt_num_rd_wgt += int(math.ceil(WL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
    chip_wgt_data_num += WL2_need[inner] # 用于生成仿真指令


    F_cur=F.copy()
    link_energy = F.copy()

    # 对core构建通信需求
    # 用到的信息: core_pkt_num_wr_opt; core_pkt_num_rd_opt; core_pkt_num_rd_wgt; core_pkt_num_rd_act
    bw_needed = (core_pkt_num_rd_act) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle 
    for item in act_core_dict:
        dst_list = act_core_dict[item]
        if if_multicast == 0:
            for dst in dst_list:
                for link in route_table[(al2_node + 1000, dst + 1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        elif if_multicast == 1:
            link_set = simple_multicast(al2_node + 1000, [dst + 1000 for dst in dst_list], route_table) 
            for link in link_set:
                F_cur[link] += ( bw_needed / bw_scales[link] )

    bw_needed = (core_pkt_num_rd_wgt) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
    for item in wgt_core_dict:
        dst_list = wgt_core_dict[item]
        if if_multicast == 0:
            for dst in dst_list:
                for link in route_table[(wl2_node + 1000, dst + 1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        elif if_multicast == 1:
            link_set = simple_multicast(wl2_node + 1000, [dst + 1000 for dst in dst_list], route_table) 
            for link in link_set:
                F_cur[link] += ( bw_needed / bw_scales[link] )


    bw_needed = (core_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
    for item in out_core_dict:
        dst_list = out_core_dict[item]
        if if_multicast == 0:
            for dst in dst_list:
                for link in route_table[(ol2_node + 1000, dst + 1000)]:
                    F_cur[link] += ( bw_needed / bw_scales[link] )
        elif if_multicast == 1:
            link_set = simple_multicast(ol2_node + 1000, [dst + 1000 for dst in dst_list], route_table) 
            for link in link_set:
                F_cur[link] += ( bw_needed / bw_scales[link] )

    bw_needed = (core_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
    for item in out_core_dict:
        dst_list = out_core_dict[item]
        for dst in dst_list:					#写output不存在多播可能
            for link in route_table[(dst + 1000, ol2_node+1000)]:
                F_cur[link] += ( bw_needed / bw_scales[link] )
    

    # 对chip构建通信需求
    dram_to_L2_F_cur = L2_to_DRAM_F_cur = 0
    bw_needed_io_die = {}
    # 用到的信息: chip_pkt_num_wr_opt; chip_pkt_num_rd_opt; chip_pkt_num_rd_wgt; chip_pkt_num_rd_act
    bw_needed = (chip_pkt_num_rd_act) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle
    bw_needed_io_die["input"] = bw_needed  
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_input/noc_bandwidth)
    
    bw_needed = (chip_pkt_num_rd_wgt) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
    bw_needed_io_die["weight"] = bw_needed  
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_weight/noc_bandwidth)

    bw_needed = (chip_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
    bw_needed_io_die["output"] = bw_needed  
    #dram_to_L2_F_cur += bw_needed / (ddr_bandwidth_io_die_output/noc_bandwidth)

    bw_needed = (chip_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
    bw_needed_io_die["output"] = bw_needed  
    #L2_to_DRAM_F_cur += bw_needed / (ddr_bandwidth_io_die_output/noc_bandwidth)
    L2_to_DRAM_F_cur = 0 # 读写复用

    bw_needed_io_die_order = sorted(bw_needed_io_die.items(), key = lambda x: x[1])
    data_type_order = [bw_needed_io_die_order[2][0], bw_needed_io_die_order[1][0], bw_needed_io_die_order[0][0], "unique"]
    for data_type in data_type_order:
        if data_type in ddr_bandwidth_io_die_method_dict:
            ddr_bandwidth_io_die_input = ddr_bandwidth_io_die_method_dict[data_type]["input"]
            ddr_bandwidth_io_die_weight = ddr_bandwidth_io_die_method_dict[data_type]["weight"]
            ddr_bandwidth_io_die_output = ddr_bandwidth_io_die_method_dict[data_type]["output"]
    
    dram_to_L2_F_cur += bw_needed_io_die["input"] / (ddr_bandwidth_io_die_input/noc_bandwidth)
    dram_to_L2_F_cur += bw_needed_io_die["weight"] / (ddr_bandwidth_io_die_weight/noc_bandwidth)
    dram_to_L2_F_cur += bw_needed_io_die["output"] / (ddr_bandwidth_io_die_output/noc_bandwidth)

    degrade_ratio = max ( max(F_cur.values()), L2_to_DRAM_F_cur, dram_to_L2_F_cur)
    if (degrade_ratio < 1):
            degrade_ratio = 1 
    #print ("F_cur",F_cur)
    #print ("degrade_ratio",degrade_ratio)

    # --------------------- 生成用于仿真的指令 ---------------------

    # 寻找最小的循环单位
    inner_cp_id = min(al1_cp_id,wl1_cp_id,ol1_cp_id,al2_cp_id,wl2_cp_id,ol2_cp_id)
    inner_cp = data_flow[inner_cp_id]
    ol1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[ol1_cp]) 
    al1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[al1_cp]) 
    wl1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[wl1_cp]) 
    ol2_repeat_interval = int(repeat_num[inner_cp] / repeat_num[ol2_cp]) 
    al2_repeat_interval = int(repeat_num[inner_cp] / repeat_num[al2_cp]) 
    wl2_repeat_interval = int(repeat_num[inner_cp] / repeat_num[wl2_cp]) 

    cal_cycle_per_run = cal_cycles[data_flow[inner_cp_id-1]]
    repeat_times = repeat_num[data_flow[inner_cp_id]]
    repeat_times_1 = repeat_num[data_flow[inner_cp_id+1]]
    #print ("ol1_repeat_interval",ol1_repeat_interval, "al1_repeat_interval",al1_repeat_interval,"wl1_repeat_interval",wl1_repeat_interval)
    #print ("ol2_repeat_interval",ol2_repeat_interval, "al2_repeat_interval",al2_repeat_interval,"wl2_repeat_interval",wl2_repeat_interval)

    # 计算最小循环单位中的packet传输数目
    core_out_packet = int(math.ceil(core_out_data_num/flit_per_pkt/neu_per_flit_psum))
    core_act_packet = int(math.ceil(core_act_data_num/flit_per_pkt/neu_per_flit_act_wgt))
    core_wgt_packet = int(math.ceil(core_wgt_data_num/flit_per_pkt/neu_per_flit_act_wgt))
    core_rd_out_packet = int (math.ceil(core_rd_out_data_num/flit_per_pkt/neu_per_flit_psum))

    chip_out_packet = int(math.ceil(chip_out_data_num/flit_per_pkt/neu_per_flit_psum))
    chip_act_packet = int(math.ceil(chip_act_data_num/flit_per_pkt/neu_per_flit_act_wgt))
    chip_wgt_packet = int(math.ceil(chip_wgt_data_num/flit_per_pkt/neu_per_flit_act_wgt))
    chip_rd_out_packet = int (math.ceil(chip_rd_out_data_num/flit_per_pkt/neu_per_flit_psum))

    # 计算插空packet数目
    core_small_wgt_packet =  round ( (core_wgt_packet  / wl1_repeat_interval) ) # 四舍五入
    core_small_act_packet =  round ( (core_act_packet  / al1_repeat_interval) ) 
    core_small_out_packet =  round ( (core_out_packet  / ol1_repeat_interval) ) 
    core_small_rd_out_packet =  round ( (core_rd_out_packet / ol1_repeat_interval) )

    chip_small_wgt_packet =  round ( (chip_wgt_packet  / wl2_repeat_interval) ) # 四舍五入
    chip_small_act_packet =  round ( (chip_act_packet  / al2_repeat_interval) ) 
    chip_small_out_packet =  round ( (chip_out_packet  / ol2_repeat_interval) ) 
    chip_small_rd_out_packet =  round ( (chip_rd_out_packet / ol2_repeat_interval) )

    # 创建文件夹
    task_dir = "./../dnn_task"
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)
    task_dir = os.path.join(task_dir,dir_name)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    os.makedirs(task_dir, exist_ok=True)
    #start_task_dir = os.path.join(task_dir,"single_engine_example_start")
    #os.makedirs(start_task_dir, exist_ok=True)
    #pipe_task_dir = os.path.join(task_dir,"single_engine_example_pipe")
    #os.makedirs(pipe_task_dir, exist_ok=True)

    pipe_task_dir = task_dir

    mem_wait_packet = {}

    for mem_id in mem_node_list:
        mem_wait_packet[mem_id] = 0

    # pipeline仿真 指令
    w_mem_file = open(pipe_task_dir+'/'+str(wl2_node)+'.txt','w')
    a_mem_file = open(pipe_task_dir+'/'+str(al2_node)+'.txt','w')
    o_mem_file = open(pipe_task_dir+'/'+str(ol2_node)+'.txt','w')

    # cc节点：wait3种数据;  cal; send output
    for cal_core_id in cc_node_list:  
        if os.path.exists(pipe_task_dir+'/'+str(cal_core_id)+'.txt'):
            core_file = open(pipe_task_dir+'/'+str(cal_core_id)+'.txt','a')
        else:
            core_file = open(pipe_task_dir+'/'+str(cal_core_id)+'.txt','w')
        
        # if os.path.exists(pipe_task_dir+'/'+str(wl2_node)+'.txt'):
        #     w_mem_file = open(pipe_task_dir+'/'+str(wl2_node)+'.txt','a')
        # else:
        #     w_mem_file = open(pipe_task_dir+'/'+str(wl2_node)+'.txt','w')
        # 
        # if os.path.exists(pipe_task_dir+'/'+str(al2_node)+'.txt'):
        #     a_mem_file = open(pipe_task_dir+'/'+str(al2_node)+'.txt','a')
        # else:
        #     a_mem_file = open(pipe_task_dir+'/'+str(al2_node)+'.txt','w')
        # 
        # if os.path.exists(pipe_task_dir+'/'+str(ol2_node)+'.txt'):
        #     o_mem_file = open(pipe_task_dir+'/'+str(ol2_node)+'.txt','a')
        # else:
        #     o_mem_file = open(pipe_task_dir+'/'+str(ol2_node)+'.txt','w')

        
        if core_small_out_packet!= 0: 
            print ("send "+str(ol2_node)+" "+str(core_small_out_packet)+" "+str(out_tag), file= core_file)
        print ("cal",cal_cycle_per_run, file = core_file)
        print ("wait "+str(core_small_act_packet) +" "+str(act_tag),file = core_file)
        print ("wait "+str(core_small_wgt_packet) +" "+str(wgt_tag),file = core_file)
        print ("wait "+str(core_small_rd_out_packet) +" "+str(out_tag),file = core_file)
        mem_wait_packet[ol2_node] += core_small_out_packet


        #with open (pipe_task_dir+'/'+str(wl2_node)+'.txt','a') as mem_file:
        if core_small_wgt_packet!= 0: print ("send "+str(cal_core_id)+" "+str(core_small_wgt_packet)+" "+str(wgt_tag), file= w_mem_file)
            
        #with open (pipe_task_dir+'/'+str(al2_node)+'.txt','a') as mem_file:
        if core_small_act_packet!= 0: print ("send "+str(cal_core_id)+" "+str(core_small_act_packet)+" "+str(act_tag), file= a_mem_file)
        
        #with open (pipe_task_dir+'/'+str(ol2_node)+'.txt','a') as mem_file:
        if core_small_rd_out_packet!= 0: print ("send "+str(cal_core_id)+" "+str(core_small_rd_out_packet)+" "+str(out_tag), file= o_mem_file)

    w_mem_file.close()
    a_mem_file.close()
    o_mem_file.close()
    core_file.close()
    
    # chiplet traffic: dram -> act L2
    for item in act_chip_dict:
        dst_list = act_chip_dict[item]
        for dst in dst_list:
            with open (pipe_task_dir+'/'+str(dst)+'.txt','a') as al2_file:
                print ("wait "+str(chip_small_act_packet) +" "+str(act_tag),file = al2_file)
            with open (pipe_task_dir+'/'+str(dram_node)+'.txt','a') as dram_file:
                print ("send "+str(dst)+" "+str(chip_small_act_packet)+" "+str(act_tag), file= dram_file)

    # chiplet traffic: dram -> wgt L2
    for item in wgt_chip_dict:
        dst_list = wgt_chip_dict[item]
        for dst in dst_list:
            with open (pipe_task_dir+'/'+str(dst)+'.txt','a') as wl2_file:
                print ("wait "+str(chip_small_wgt_packet) +" "+str(wgt_tag),file = wl2_file)
            with open (pipe_task_dir+'/'+str(dram_node)+'.txt','a') as dram_file:
                print ("send "+str(dst)+" "+str(chip_small_wgt_packet)+" "+str(wgt_tag), file= dram_file)

    # chiplet traffic: dram -> out L2 (send part)
    for item in out_chip_dict:
        dst_list = out_chip_dict[item]
        for dst in dst_list:
            with open (pipe_task_dir+'/'+str(dram_node)+'.txt','a') as dram_file:
                print ("send "+str(dst)+" "+str(chip_small_rd_out_packet)+" "+str(out_tag), file= dram_file)

    # chiplet traffic: out L2 -> dram (send part)
    for item in out_chip_dict:
        dst_list = out_chip_dict[item]
        for dst in dst_list:
            with open (pipe_task_dir+'/'+str(dst)+'.txt','a') as ol2_file:
                print ("send "+str(dram_node)+" "+str(chip_small_out_packet) +" "+str(out_tag),file = ol2_file)
            mem_wait_packet[dram_node] += chip_small_out_packet

    # chiplet traffic: out L2 -> dram (wait part)
    with open (pipe_task_dir+'/'+str(dram_node)+'.txt','a') as dram_file:
        print ("wait "+str(mem_wait_packet[dram_node])+" "+str(out_tag), file= dram_file)

    # chiplet traffic: dram -> out L2 (wait part)
    for item in out_chip_dict:
        dst_list = out_chip_dict[item]
        for dst in dst_list:
            with open (pipe_task_dir+'/'+str(dst)+'.txt','a') as ol2_file:
                print ("wait "+str(chip_small_rd_out_packet) +" "+str(out_tag),file = ol2_file)

    # core traffic ol2 node task (wait part)
    with open (pipe_task_dir+'/'+str(ol2_node)+'.txt','a') as mem_file:
        if mem_wait_packet[ol2_node] != 0: print ("wait "+str(mem_wait_packet[ol2_node])+" "+str(out_tag), file= mem_file)
    # all finish

    for sim_node in range (all_sim_node_num):   
        with open (pipe_task_dir+'/'+str(sim_node)+'.txt','a') as node_file:
            print ("finish",file = node_file)
    # 启动延迟仿真 指令
    
    # communication pair pattern
    trace_file = open(pipe_task_dir+'/comm_pair.txt','w')
    # -- al2_node -> c_node
    print("#--- act_node to c_node ---#", file = trace_file)
    print("src : dst\t\t\t\tpacket_num", file = trace_file)
    for i, items in act_core_dict.items():
        print("({}):({})\t\t{}".format(al2_node, str(items).replace("[",'').replace("]",''), core_small_act_packet), file = trace_file)
    print("#--- wgt_node to c_node ---#", file = trace_file)
    for i, items in wgt_core_dict.items():
        print("({}):({})\t\t{}".format(wl2_node, str(items).replace("[",'').replace("]",''), core_small_wgt_packet), file = trace_file)
    print("#--- out_node to c_node ---#", file = trace_file)
    for i, items in out_core_dict.items():
        print("({}):({})\t\t{}".format(ol2_node, str(items).replace("[",'').replace("]",''), core_small_out_packet), file = trace_file)
        
    print("#--- c_node to out_node ---#", file = trace_file)
    for i, items in out_core_dict.items():
        print("({}):({})\t\t{}".format(str(items).replace("[",'').replace("]",''), ol2_node, core_small_rd_out_packet), file = trace_file)
    trace_file.close()
    #print("act_core_num = {};\nact_core_dict: {}".format(core_small_act_packet, act_core_dict), file = trace_file)
    #print("wgt_core_dict: num {}, {}".format(core_small_wgt_packet, wgt_core_dict), file = trace_file)
    #print("out_core_dict: num {}, {}".format(core_small_out_packet, out_core_dict), file = trace_file)
    #print("rd_out_core_dict: num {}, {}".format(core_small_rd_out_packet, out_core_dict))

    return repeat_times, repeat_times_1