import math
import os
import random
import copy
from single_engine_predict_intralayer import *
from mesh_hetero import *
from config import *
import argparse
from basicParam_noc_nop import *
from gaTest_noc_nop import *

class Encoder:
    def __init__(self, EncodeType, network_param, spatial_parallel = None, debug=0, architecture_level=3, temporal_level=3, debug_file="./random_test_record.txt"):
        self.network_param = network_param
        self.spatial_parallel = spatial_parallel
        self.architecture_level = architecture_level
        self.architecture_level_list = ["pe", "chiplet", "package"]
        self.temporal_level = temporal_level
        self.debug = debug
        self.debug_file = debug_file

        self.temporal_param = None
        #self.getTemporalParam()

        self.EncodeType = EncodeType
        self.tile_num_max = 40
        self.loop_tile_dict = {}
        #self.getTileDict()
        self.tile_order_dict = None
        self.getTileOrder()
    
    def getTileOrder(self):
        list_t = list(range(self.temporal_level))
        self.tile_order_dict = [list_t]
        order_num = 1
        for i in range(self.temporal_level):
            order_num *= (i+1)
        print("order_num: ", order_num)
        while (len(self.tile_order_dict) < order_num):
            list_shuffle = copy.deepcopy(list_t)
            random.shuffle(list_shuffle)
            if list_shuffle not in self.tile_order_dict:
                self.tile_order_dict.append(list_shuffle)
        #print("self.tile_order_dict:", self.tile_order_dict)
        #exit()

    def getTileDict(self):
        for dim in self.temporal_param:
            if dim != "R" and dim != "S":
                tile_num = 0
                iter_num = 0
                self.loop_tile_dict[dim] = []
                num = self.temporal_param[dim]
                while (tile_num < self.tile_num_max and iter_num < 500):
                    iter_num += 1
                    tile_list_all = setPartition_1(num, self.temporal_level)
                    tile_list = tile_list_all[0: self.temporal_level]
                    tile_list.sort(reverse = True)
                    if tile_list not in self.loop_tile_dict[dim]:
                        self.loop_tile_dict[dim].append(tile_list)
                        tile_num += 1

    def getTemporalParam(self):
        self.temporal_param = {}
        for dim in self.network_param:
            if dim == "P" or dim == "Q" or dim == "C" or dim == "K" or dim == "R"or dim == "S":
                self.temporal_param[dim] = self.network_param[dim]
                for level in range(self.architecture_level):
                    level_name = self.architecture_level_list[level]
                    # print("level_name = ", level_name)
                    self.temporal_param[dim] = math.ceil(self.temporal_param[dim] / self.spatial_parallel[level_name][dim])
    
    def setSpatialParallel(self, spatialParallel, loop_tile_dict = None):
        self.spatial_parallel = spatialParallel
        self.getTemporalParam()
        if loop_tile_dict == None:
            self.getTileDict()
        else:
            self.loop_tile_dict = copy.deepcopy(loop_tile_dict)

    def getTempPartitonCode(self):
        #---维度拆分（时间并行）---

        ran1 = random.randint(0,2)
        if ran1 == 0:
            # ---- 质因数分解
            Pset = setPartition_1(self.temporal_param["P"], self.temporal_level)
            Qset = setPartition_1(self.temporal_param["Q"], self.temporal_level)
            Cset = setPartition_1(self.temporal_param["C"], self.temporal_level)
            Kset = setPartition_1(self.temporal_param["K"], self.temporal_level)
        else:
            Pset = setPartition(self.temporal_param["P"], self.temporal_level)
            Qset = setPartition(self.temporal_param["Q"], self.temporal_level)
            Cset = setPartition(self.temporal_param["C"], self.temporal_level)
            Kset = setPartition(self.temporal_param["K"], self.temporal_level)
        
        partition_code = Pset[0:self.temporal_level-1] + Qset[0:self.temporal_level-1] + Cset[0:self.temporal_level-1] + Kset[0:self.temporal_level-1]

        return partition_code
    
    def getTempOrderCode(self):
        #---维度排序（时间并行）---
        order_code = []

        for level in range(self.temporal_level):
            for _ in range(architecture_dim_num[level]):
                priority = random.random()
                order_code.append(priority)
        
        return order_code

    def getCode_index(self):
        # -- loop tile
        partition_code = []
        PE_parallel_order = random.randint(0,1)
        chiplet_parallel_order = random.randint(0,1)
        partition_code.append(PE_parallel_order)
        partition_code.append(chiplet_parallel_order)
        tile_order_num = len(self.tile_order_dict)
        for dim in self.loop_tile_dict:
            tile_num = len(self.loop_tile_dict[dim])
            index = random.randint(0,tile_num-1)
            tile_order_index = random.randint(0,tile_order_num-1)
            partition_code.append(index)
            partition_code.append(tile_order_index)
        
        # -- loop order
        order_code = self.getTempOrderCode()

        code = partition_code + order_code

        return code

    def CodeChange_index(self, code):
        # - spatial for
        parallel_dict = {0:[], 1:[]}
        parallel_dim_dict = {0:[], 1:[]}
        parallel_order = code[0:2]
        for level in range(self.architecture_level):
            if level > 0:
                level_name = self.architecture_level_list[level]
                for dim in self.spatial_parallel[level_name]:
                    dim_id = dim2id[dim]
                    num = self.spatial_parallel[level_name][dim]
                    if num > 1:
                        parallel_dict[level-1].append(num)
                        parallel_dim_dict[level-1].append(dim_id)
                #print("parallel_dict[level-1] : ", parallel_dict[level-1])
                #print("parallel_dim_dict[level-1] : ", parallel_dim_dict[level-1])
                if len(parallel_dict[level-1]) == 1:
                    parallel_dict[level-1].append(1)
                    parallel_dim_dict[level-1].append(0)
                elif len(parallel_dict[level-1]) == 0:
                    parallel_dict[level-1].append(1)
                    parallel_dim_dict[level-1].append(0)
                    parallel_dict[level-1].append(1)
                    parallel_dim_dict[level-1].append(0)
        
        for i in range(len(parallel_order)):
            order = parallel_order[i]
            if order == 1:
                # -- reverse
                parallel_dict[i][0], parallel_dict[i][1] = parallel_dict[i][1], parallel_dict[i][0]
                parallel_dim_dict[i][0], parallel_dim_dict[i][1] = parallel_dim_dict[i][1], parallel_dim_dict[i][0]
        
        parallel_code_c = parallel_dim_dict[0] + parallel_dim_dict[1] + parallel_dict[0] + parallel_dict[1]
        
        # - temporal
        P_tile_index = code[0+2]
        P_tile_order = code[1+2]
        Q_tile_index = code[2+2]
        Q_tile_order = code[3+2]
        C_tile_index = code[4+2]
        C_tile_order = code[5+2]
        K_tile_index = code[6+2]
        K_tile_order = code[7+2]
        order_code = code[8+2:]
        # -- temporal for code
        for_code = {}
        for_code[0] = [self.loop_tile_dict["P"][P_tile_index][i] for i in self.tile_order_dict[P_tile_order]]
        for_code[1] = [self.loop_tile_dict["Q"][Q_tile_index][i] for i in self.tile_order_dict[Q_tile_order]]
        for_code[2] = [self.loop_tile_dict["C"][C_tile_index][i] for i in self.tile_order_dict[C_tile_order]]
        for_code[3] = [self.loop_tile_dict["K"][K_tile_index][i] for i in self.tile_order_dict[K_tile_order]]

        partition_code_c = for_code[0] + for_code[1] + for_code[2] + for_code[3]

        # -- loop order
        loop_order_c = []
        t_start = 0
        for t_level in range(self.temporal_level):
            t_num = architecture_dim_num[t_level]
            loop_order = copy.deepcopy(order_code[t_start: t_start+t_num])
            t_start += t_num

            m_sorted = sorted(enumerate(loop_order), key=lambda x:x[1])
            loop_order_sorted_index = [m[0] for m in m_sorted]
            loop_order_c += copy.deepcopy(loop_order_sorted_index)

        code_c = parallel_code_c + partition_code_c + loop_order_c

        if self.debug == 1:
            print("Debug in CodeChange :")
            print("--- code_change : ", code_c)

        return code_c

    def getCode_num(self):

        pe_parallel_order, chiplet_parallel_order = random.randint(0,1), random.randint(0,1)
        parallel_order_code = [pe_parallel_order, chiplet_parallel_order]


        partition_code = self.getTempPartitonCode()
        order_code = self.getTempOrderCode()

        code = parallel_order_code + partition_code + order_code

        return code
    
    def CodeChange_num(self, code):
        # - spatial for
        parallel_dict = {0:[], 1:[]}
        parallel_dim_dict = {0:[], 1:[]}
        for level in range(self.architecture_level):
            if level > 0:
                level_name = self.architecture_level_list[level]
                for dim in self.spatial_parallel[level_name]:
                    dim_id = dim2id[dim]
                    num = self.spatial_parallel[level_name][dim]
                    if num > 1:
                        parallel_dict[level-1].append(num)
                        parallel_dim_dict[level-1].append(dim_id)
                if len(parallel_dict[level-1]) == 1:
                    parallel_dict[level-1].append(1)
                    parallel_dim_dict[level-1].append(0)
                elif len(parallel_dict[level-1]) == 0:
                    parallel_dict[level-1].append(1)
                    parallel_dim_dict[level-1].append(0)
                    parallel_dict[level-1].append(1)
                    parallel_dim_dict[level-1].append(0)
        
        parallel_order = code[0:2]
        for i in range(len(parallel_order)):
            order = parallel_order[i]
            if order == 1:
                # -- reverse
                parallel_dict[i][0], parallel_dict[i][1] = parallel_dict[i][1], parallel_dict[i][0]
                parallel_dim_dict[i][0], parallel_dim_dict[i][1] = parallel_dim_dict[i][1], parallel_dim_dict[i][0]

        parallel_code_c = parallel_dim_dict[0] + parallel_dim_dict[1] + parallel_dict[0] + parallel_dict[1]
        
        # - temporal
        partition_code = code[0+2:8+2]
        order_code = code[8+2:]
        # -- temporal for code
        for_code = {}
        P_rest = self.temporal_param["P"]
        Q_rest = self.temporal_param["Q"]
        C_rest = self.temporal_param["C"]
        K_rest = self.temporal_param["K"]
        for t_level in range(self.temporal_level-1):
            for_p = partition_code[t_level]
            for_q = partition_code[(self.temporal_level-1)*1 + t_level]
            for_c = partition_code[(self.temporal_level-1)*2 + t_level]
            for_k = partition_code[(self.temporal_level-1)*3 + t_level]
            for_code[0].append(for_p)
            for_code[1].append(for_q)
            for_code[2].append(for_c)
            for_code[3].append(for_k)

            P_rest /= for_p
            Q_rest /= for_q
            C_rest /= for_c
            K_rest /= for_k
        for_code[0].append(math.ceil(P_rest))
        for_code[1].append(math.ceil(Q_rest))
        for_code[2].append(math.ceil(C_rest))
        for_code[3].append(math.ceil(K_rest))
        
        partition_code_c = for_code[0] + for_code[1] + for_code[2] + for_code[3]

        # -- loop order
        loop_order_c = []
        t_start = 0
        for t_level in range(self.temporal_level):
            t_num = architecture_dim_num[t_level]
            loop_order = copy.deepcopy(order_code[t_start: t_start+t_num])
            t_start += t_num

            m_sorted = sorted(enumerate(loop_order), key=lambda x:x[1])
            loop_order_sorted_index = [m[0] for m in m_sorted]
            loop_order_c += copy.deepcopy(loop_order_sorted_index)
        
        code_c = parallel_code_c + partition_code_c + loop_order_c

        return code_c

    def getCode(self):
        if self.EncodeType == "num":
            code = self.getCode_num()
        elif self.EncodeType == "index":
            code = self.getCode_index()
        else:
            print("Error EncodeType({}), (num,index) are provided".format(self.EncodeType))
            exit()
        return code
    
    def codeChange(self, code):
        if self.EncodeType == "num":
            code_c = self.CodeChange_num(code)
        elif self.EncodeType == "index":
            code_c = self.CodeChange_index(code)
        else:
            print("Error EncodeType({}), (num,index) are provided".format(self.EncodeType))
            exit()
        return code_c

class Decoder:
    def __init__(self, temporal_level, HW_param, nn_param, temporal_order_mergy=0):
        self.temporal_level = temporal_level
        self.temporal_order_mergy = temporal_order_mergy 	# 多级的order是否要合并考虑先后顺序，还是按各层级各自考虑

        self.HW_param = HW_param
        self.PE_param = HW_param["PE"]
        self.PEs = self.PE_param[0] * self.PE_param[1]
        self.chiplet_param = HW_param["Chiplet"]
        self.chiplets = self.chiplet_param[0] * self.chiplet_param[1]
        self.R = nn_param["R"]
        self.S = nn_param["S"]

        self.NoC_node_offset = []
        self.NoP2NoCnode = []
        self.A_W_offset = {}
        self.setNodeID()
    
    def setNodeID(self):
        Chiplet_lenth = self.chiplet_param[0]
        Chiplet_height = self.chiplet_param[1]
        PE_lenth = self.PE_param[0]
        PE_height = self.PE_param[1]
        assert(Chiplet_lenth*Chiplet_height == self.chiplets)
        assert(PE_lenth*PE_height == self.PEs)

        if PE_height == 2:
            self.A_W_offset["o"] = 0
            self.A_W_offset["a"] = PE_lenth + 1
            self.A_W_offset["w"] = PE_lenth + 1
            self.A_W_offset["noc-chiplet"] = 0
        else:
            assert(PE_height > 1)
            self.A_W_offset["o"] = 0
            self.A_W_offset["a"] = PE_lenth + 1
            self.A_W_offset["w"] = (PE_lenth + 1) * 2
            self.A_W_offset["noc-chiplet"] = 0
        PE_num = (PE_lenth + 1) * PE_height

        num = 0
        for i in range(self.chiplets):
            if i % Chiplet_lenth == 0: 
                self.NoP2NoCnode.append(0)
            num += PE_num
            self.NoC_node_offset.append(num)
            self.NoP2NoCnode.append(num)

            if (i+1) % Chiplet_lenth == 0:
                num += PE_num

    # 获得并行下的节点分组情况
    def setmappingSet(self, height, lenth, set1, set2, ol2_node = A_W_offset['o']):
        num = height * lenth
        assert(num >= set1*set2)
        list1 = {}
        list2 = {}
        node_list = []
        ID = 0
        for i in range(num):
            if i % lenth == ol2_node:
                ID += 1
            node_list.append(ID)
            ID += 1
        for i in range(set1*set2):
            set1_id = i // set2
            if set1_id not in list1:
                list1[set1_id] = []
            list1[set1_id].append(node_list[i])

        for i in range(set1*set2):
            set2_id = i // set1
            if set2_id not in list2:
                list2[set2_id] = []
            list2[set2_id].append(list1[i % set1][set2_id])
        return list1, list2

    def getActWgtSet(self, correlate_p1, correlate_p2, p1_set, p2_set):
        dict = {}
        if correlate_p1 == 0 and correlate_p2 == 0:
            dict[0] = []
            for set_id_p1 in p1_set:
                dict[0] += p1_set[set_id_p1]
        elif correlate_p1 == 1 and correlate_p2 == 0:
            dict = p1_set
        elif correlate_p1 == 0 and correlate_p2 == 1:
            dict = p2_set
        else:
            node_num = 0
            for set_id_p1 in p1_set:
                for node_id in p1_set[set_id_p1]:
                    dict[node_num] = [node_id]
                    node_num += 1
        return dict
    
    # 获得计算核心对于act与wgt的共享情况
    # p1_num, p2_num分别是并行维度1和2的数目
    def getPEDistribution(self, flag, height, lenth, act_correlate, wgt_correlate, parallel_num):
        act_PE_dict = {}
        wgt_PE_dict = {}
        act_set_type = "0"
        wgt_set_type = "0"

        #---act + wgt send节点列表---
        if flag == 0:
            act_PE_dict["send"] = {0:[A_W_offset["a"]]}
            wgt_PE_dict["send"] = {0:[A_W_offset["w"]]}
        else:
            act_PE_dict["send"] = {0:[0]}
            wgt_PE_dict["send"] = {0:[0]}

        #---获得p1，p2的列表
        if len(parallel_num) == 0:
            p1_num = 1
            p2_num = 1
        elif len(parallel_num) == 1:
            p1_num = parallel_num[0]
            p2_num = 1
        else:
            p1_num = parallel_num[0]
            p2_num = parallel_num[1]
        p1_dict, p2_dict = self.setmappingSet(height, lenth, p1_num, p2_num)
        #---act recv节点列表---
        if len(act_correlate) == 0:
            act_p1 = 0
            act_p2 = 0
            wgt_p1 = 0
            wgt_p2 = 0
        elif len(act_correlate) == 1:
            act_p1 = act_correlate[0]
            act_p2 = 0
            wgt_p1 = wgt_correlate[0]
            wgt_p2 = 0
        else:
            act_p1 = act_correlate[0]
            act_p2 = act_correlate[1]
            wgt_p1 = wgt_correlate[0]
            wgt_p2 = wgt_correlate[1]

        act_PE_dict["recv"] = self.getActWgtSet(act_p1, act_p2, p1_dict, p2_dict)
        wgt_PE_dict["recv"] = self.getActWgtSet(wgt_p1, wgt_p2, p1_dict, p2_dict)

        act_set_num = len(act_PE_dict["recv"])
        wgt_set_num = len(wgt_PE_dict["recv"])
        
        act_set_type = "set_"+str(act_set_num)+"e_"+str(int(height * lenth / act_set_num))
        wgt_set_type = "set_"+str(wgt_set_num)+"e_"+str(int(height * lenth / wgt_set_num))
        return act_PE_dict, wgt_PE_dict, act_set_type, wgt_set_type

    # 给dict的每个元素加上固定偏移量num
    # dict = {"send":{0:[],...},"recv":{0:[]...}}
    def dictAddInt(self, dict, num):
        send = dict["send"]
        recv = dict["recv"]
        for i in send:
            for x in range(len(send[i])):
                send[i][x] += num

        for i in recv:
            for x in range(len(recv[i])):
                recv[i][x] +=  num
        dict1 = copy.deepcopy({"send":send,"recv":recv})
        return dict1

    # 转换为chiplet
    def dictChipletChange(self, dict, flag1, flag2):
        send = dict["send"]
        recv = dict["recv"]
        for i in send:
            for x in range(len(send[i])):
                num = send[i][x]
                send[i][x] = self.NoP2NoCnode[num] + self.A_W_offset[flag1]
        for i in recv:
            for x in range(len(recv[i])):
                num = recv[i][x]
                recv[i][x] = self.NoP2NoCnode[num] + self.A_W_offset[flag2]
        dict1 = copy.deepcopy({"send":send,"recv":recv})
        return dict1

    # type = 0 : NoC ; type = 1 : NoP
    def getPEExtent(self, dict, type=0, flag1 = 0, flag2 = 0):
        list = []
        #list.append(dict)
        if type == 0:
            for i in self.NoC_node_offset:
                dict1 = copy.deepcopy(dict)
                dict1 = self.dictAddInt(dict1, i)		
                list.append(dict1)
        else:
            dict1 = copy.deepcopy(dict)
            dict1 = self.dictChipletChange(dict1,flag1,flag2)
            return dict1
        return list

    # 获得输出特征图的数据节点通信关系
    def getOutputDict(self, runtimeCoreNum, runtimeChipNum):
        rd_out_PE_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.PE_param[0], self.PE_param[1], [1,1], [1,1], [runtimeCoreNum, 1])
        wr_out_PE_dict_temp = {"send":rd_out_PE_dict_temp["recv"],"recv":{0:[0]}}
        rd_out_Chiplet_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.chiplet_param[0], self.chiplet_param[1], [1,1], [1,1], [runtimeChipNum, 1])
        wr_out_Chiplet_dict_temp = {"send":rd_out_Chiplet_dict_temp["recv"],"recv":{0:[0]}}

        #if self.PEs == 16:
        #	rd_out_PE_dict_temp = {"send":{0:[0]},"recv":set_16_e_1[0]}
        #	wr_out_PE_dict_temp = {"send":set_16_e_1[0],"recv":{0:[0]}}
        #elif self.PEs == 4:
        #	rd_out_PE_dict_temp = {"send":{0:[0]},"recv":set_4_e_1[0]}
        #	wr_out_PE_dict_temp = {"send":set_4_e_1[0],"recv":{0:[0]}}
        #if self.Chiplets == 16:
        #	rd_out_Chiplet_dict_temp = {"send":{0:[0]},"recv":set_16_e_1[0]}
        #	wr_out_Chiplet_dict_temp = {"send":set_16_e_1[0],"recv":{0:[0]}}
        #elif self.Chiplets == 4:
        #	rd_out_Chiplet_dict_temp = {"send":{0:[0]},"recv":set_4_e_1[0]}
        #	wr_out_Chiplet_dict_temp = {"send":set_4_e_1[0],"recv":{0:[0]}}
        rd_out_PE_dict = self.getPEExtent(rd_out_PE_dict_temp)
        wr_out_PE_dict = self.getPEExtent(wr_out_PE_dict_temp)
        rd_out_Chiplet_dict = self.getPEExtent(rd_out_Chiplet_dict_temp,1,"o","o")
        wr_out_Chiplet_dict = self.getPEExtent(wr_out_Chiplet_dict_temp,1,"o","o")
        return rd_out_PE_dict, wr_out_PE_dict, rd_out_Chiplet_dict, wr_out_Chiplet_dict

    def decode(self, code):
        # code --> for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list
        spatial_parallel_dim = code[0:4]							# --> 0:P, 1:Q, 2:C, 3:K
        spatial_parallel_num = code[4:8]
        temporal_parallel_num = {}
        temporal_parallel_num[0] = code[8:(8+1*self.temporal_level)]
        temporal_parallel_num[1] = code[(8+1*self.temporal_level):(8+2*self.temporal_level)]
        temporal_parallel_num[2] = code[(8+2*self.temporal_level):(8+3*self.temporal_level)]
        temporal_parallel_num[3] = code[(8+3*self.temporal_level):(8+4*self.temporal_level)]
        temporal_parallel_order = code[(8+4*self.temporal_level):]	# --> 6, 4, ...

        # --> temporal parallel decode
        dataflow = []	# --> ['P1', 'Q1', 'P2'..., 'top']
        ol1_ratio = []
        al1_ratio = []
        wl1_ratio = []
        all_param = []
        out_final = []
        out_correlation_final = 0
        dim_t_list = [1,1,1,1,1,1]
        if self.temporal_order_mergy:
            # todo
            pass
        else:
            for i in range(self.temporal_level):
                if i == 0:
                    order = temporal_parallel_order[0:6]
                else:
                    order = temporal_parallel_order[2+i*4 : 6+i*4]
                
                for dim_id in order:
                    t_index = dim_t_list[dim_id]
                    dim = dim_list[dim_id] + str(t_index)
                    if dim_id == 4:
                        num = self.R
                    elif dim_id == 5:
                        num = self.S
                    else:
                        num = temporal_parallel_num[dim_id][t_index-1]
                    if num == 1:
                        pass
                    else:
                        dataflow.append(dim)
                        all_param.append(num)
                        if O_correlation[dim_id]:
                            ol1_ratio.append(num)
                            out_correlation_final = len(dataflow)
                        else:
                            ol1_ratio.append(1)
                        if A_correlation[dim_id]:
                            al1_ratio.append(num)
                        else:
                            al1_ratio.append(1)
                        if W_correlation[dim_id]:
                            wl1_ratio.append(num)
                        else:
                            wl1_ratio.append(1) 
                    
                    dim_t_list[dim_id] += 1
        
        for i in range(len(ol1_ratio)):
            if i < out_correlation_final:
                out_final.append(0)
            else:
                out_final.append(1)

        dataflow.append("top")
        ol1_ratio.append(1)
        al1_ratio.append(1)
        wl1_ratio.append(1)
        all_param.append(1)
        out_final.append(1)
        
        temporal_for_list = {}
        temporal_for_list[0] = dataflow
        temporal_for_list[1] = ol1_ratio
        temporal_for_list[2] = al1_ratio
        temporal_for_list[3] = wl1_ratio
        temporal_for_list[4] = all_param
        temporal_for_list[5] = out_final
        temporal_for_list[6] = None
        temporal_for_list[7] = None
        temporal_for_list[8] = None
        temporal_for_list[9] = None

        partition_list = {"P": temporal_parallel_num[0], "Q": temporal_parallel_num[1], "C": temporal_parallel_num[2], "K": temporal_parallel_num[3]}
        for i in range(3-self.temporal_level):
            partition_list["P"].append(1)
            partition_list["Q"].append(1)
            partition_list["C"].append(1)
            partition_list["K"].append(1)
        # --> spatial parallel decode
        parallel_dim_list = {0:[1,1,1,1],1:[1,1,1,1]}
        act_wgt_dict = {}
        act_correlate = {"chiplet":[], "PE":[]}
        wgt_correlate = {"chiplet":[], "PE":[]}
        parallel_set_dict = {"chiplet":[], "PE":[]}
        for i, dim_id in enumerate(spatial_parallel_dim):
            dim_num = spatial_parallel_num[i]
            if i < 2:
                archi = "PE"
                parallel_dim_list[0][dim_id] *= dim_num
            else:
                archi = "chiplet"
                parallel_dim_list[1][dim_id] *= dim_num
            
            if dim_num == 1:
                pass
            else:
                act_correlate[archi].append(A_correlation[dim_id])
                wgt_correlate[archi].append(W_correlation[dim_id])
                parallel_set_dict[archi].append(dim_num)
        
        act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE, if_wgt_share_PE = self.getPEDistribution(0, self.PE_param[0], self.PE_param[1], act_correlate["PE"],wgt_correlate["PE"], parallel_set_dict["PE"])
        act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet, if_wgt_share_Chiplet = self.getPEDistribution(1, self.chiplet_param[0], self.chiplet_param[1], act_correlate["chiplet"],wgt_correlate["chiplet"], parallel_set_dict["chiplet"])
        act_wgt_dict["act_core"] = self.getPEExtent(act_PE_dict_temp)
        act_wgt_dict["wgt_core"] = self.getPEExtent(wgt_PE_dict_temp)
        act_wgt_dict["act_chiplet"] = self.getPEExtent(act_Chiplet_dict_temp,1,"o","a")
        act_wgt_dict["wgt_chiplet"] = self.getPEExtent(wgt_Chiplet_dict_temp,1,"o","w")

        out_dict = {}
        runtimeCoreNum = 1
        runtimeChipNum = 1
        for mm in range(4):
            runtimeCoreNum *= parallel_dim_list[0][mm]
            runtimeChipNum *= parallel_dim_list[1][mm]
        out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = self.getOutputDict(runtimeCoreNum, runtimeChipNum)

        return temporal_for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, code


def randomTest(CodeGen, Parser, iterTimes, spatial_parallel_list, memory_param, NoC_param, all_sim_node_num , if_multicast, excel_filename, workload_name, i_act_enough, weight_enough, fuse_par_num, fuse_tag, objective="edp", architecture = "mesh_DDR4", io_die_tag = 1):
    degrade_ratio_list = []
    excel_datas = []

    best_fitness = 0
    fitness_list = []
    best_fitness_list = []
    best_sample = {}
    best_neu_needed = {}
    best_pkt_needed = {}

    energy_list_d = {}
    latency_list_d = {}
    sp_id = 0

    iters_per_sp = math.ceil(iterTimes / len(spatial_parallel_list))
    for spatial_parallel in spatial_parallel_list:
        sp_energy_list = []
        sp_latency_list = []
        CodeGen.setSpatialParallel(spatial_parallel)
        print("SpatialParallel is {} ----------".format(spatial_parallel))
        for i in range(iters_per_sp):
            print("iterTime----({}, {}, {})".format(i, iters_per_sp, iterTimes))
            # --- 生成个代，与个体解码 --- #
            code = CodeGen.getCode()
            code_c = CodeGen.codeChange(code)
            for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, code = Parser.decode(code_c)
            # --- 计算适应度 --- #
            delay, degrade_ratio, degrade_ratio_dict, _, pkt_needed, neu_needed, compuation_cycles, runtime_list, cp_list, utilization_ratio_list, energy_dram_list, energy_L2_list, energy_L1_list, energy_die2die, energy_MAC, energy_psum_list, delay_psum, worstlinks = \
                calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, CodeGen.network_param, Parser.HW_param, memory_param, NoC_param, if_multicast, architecture, i_act_enough, 0, weight_enough, fuse_par_num, fuse_tag, io_die_tag = io_die_tag)
            
            
            #---比较适应度，并记录相关变量---
            e_mem = sum(energy_dram_list)+sum(energy_L2_list)+sum(energy_L1_list)
            e_sum = e_mem + energy_die2die + energy_MAC + energy_psum_list[2]
            edp = (delay + delay_psum) * e_sum  /(PE_freq * freq_1G) # pJ*s
            sp_energy_list.append(e_sum)
            sp_latency_list.append(delay + delay_psum)

            if objective == "edp":
                fitness = edp
            elif objective == "energy":
                fitness = e_sum
            elif objective == "latency":
                fitness = delay + delay_psum
            if best_fitness == 0 or fitness < best_fitness:
                best_fitness = fitness
                best_compuation_cycles = compuation_cycles
                best_degrade_ratio = degrade_ratio
                best_degrade_ratio_dict = copy.deepcopy(degrade_ratio_dict)
                best_code = code
                best_sample["for_list"] 			= copy.deepcopy(for_list)
                best_sample["act_wgt_dict"] 		= copy.deepcopy(act_wgt_dict)
                best_sample["out_dict"] 			= copy.deepcopy(out_dict)
                best_sample["parallel_dim_list"] 	= copy.deepcopy(parallel_dim_list)
                best_sample["partition_list"] 		= copy.deepcopy(partition_list)
                best_sample["fitness"] 				= fitness
                best_sample["edp"] 					= edp
                best_sample["energy"] 				= e_sum
                best_sample["delay"] 				= delay + delay_psum
                best_sample["compuation_cycles"] 	= compuation_cycles
                best_sample["degrade_ratio"] 		= degrade_ratio
                best_sample["code"] 				= code
                best_pkt_needed				= copy.deepcopy(pkt_needed)
                best_neu_needed				= copy.deepcopy(neu_needed)

            fitness_list.append(fitness)
            best_fitness_list.append(best_fitness)
            degrade_ratio_list.append (degrade_ratio_dict)
            
            print("fitness_min = {}, compuation_cycles = {}, degrade_ratio = {}".format(best_fitness, best_compuation_cycles, str(best_degrade_ratio_dict)))
            
        energy_list_d[sp_id] = sp_energy_list
        latency_list_d[sp_id] = sp_latency_list
        sp_id += 1
    #---生成task file
    task_repeat_nums = createTaskFile(workload_name, best_pkt_needed, best_neu_needed, best_sample["for_list"], best_sample["act_wgt_dict"], best_sample["out_dict"], best_sample["parallel_dim_list"], best_sample["partition_list"],CodeGen.network_param, Parser.HW_param, memory_param, NoC_param, all_sim_node_num, if_multicast, architecture, io_die_tag)

    return best_sample, best_degrade_ratio_dict, latency_list_d, energy_list_d, task_repeat_nums


def getLayerParam(app_name):
    abs_path = abs_path = os.path.dirname(os.path.abspath(__file__))
    f = open(abs_path + "/nn_input_noc_nop/" + app_name + ".txt")

    print("network model ----- " + app_name + " -------------")

    lines = f.readlines()
    layer_dict = {}
    layer_name_list = []
    layer_num = 0
    for line in lines:
        if line.startswith("#") or line.startswith("*"):
            pass
        else:
            line = line.replace("\n","")
            line_item = line.split(" ")
            layer_name = line_item[0] + "-" + str(layer_num)
            H = int(line_item[1])
            M = int(line_item[2])
            P = int(line_item[8])
            Q = int(line_item[9])
            C = int(line_item[3])
            K = int(line_item[7])
            R = int(line_item[4])
            S = int(line_item[4])
            stride = int(line_item[5])
            padding = int(line_item[6])
            layer = {"P":P,"Q":Q,"C":C,"K":K,"R":R,"S":S, "stride":stride, "padding":padding, "iact_num":H*M*C, "oact_num":P*Q*K, "w_num":K*C*R*S}
            layer_dict[layer_num] = layer
            layer_name_list.append(layer_name)
            layer_num += 1
    
    layer_list = []
    layer_dict_unique = {}
    layer_name_dict = {}
    layer_num -= 1
    for i, layer in layer_dict.items():
        if i == 0 and i == layer_num:
            partition_tag = 0
            weight_behind = 0
            weight_above = 0
        elif i == 0:
            partition_tag = 0
            weight_behind = layer_dict[i+1]["w_num"]
            weight_above = 0
        elif i == layer_num:
            partition_tag = 2
            weight_behind = 0
            weight_above = layer_dict[i-1]["w_num"]
        else:
            partition_tag = 1
            weight_behind = layer_dict[i+1]["w_num"]
            weight_above = layer_dict[i-1]["w_num"]
        
        if i == layer_num:
            partition_add = 0
        else:
            stride_behind = layer_dict[i+1]["stride"]
            padding_behind = layer_dict[i+1]["padding"]
            R_behind = layer_dict[i+1]["R"]
            partition_add = R_behind - stride_behind - padding_behind
        
        layer["partition"] = [partition_tag, partition_add]
        layer["w_num_behind"] = weight_behind
        layer["w_num_above"] = weight_above
        layer_name = layer_name_list[i]

        print(str(layer_name) + " : " + str(layer))

        if layer not in layer_list:
            layer_dict_unique[layer_name] = layer
            layer_list.append(layer)
            layer_name_dict[layer_name] = layer_name
        else:
            for layer_name_1, layer_1 in layer_dict_unique.items():
                if layer == layer_1:
                    layer_name_same = layer_name_1
                    break
            layer_name_dict[layer_name] = layer_name_same
    
    for i in range(len(layer_name_list)):
        layer_name = layer_name_list[i]
        if layer_name in layer_dict_unique:
            if i < len(layer_name_list)-1:
                i_t = i + 1
                layer_name_tail = layer_name
                while layer_name_tail in layer_name_list[0:i+1]:
                    layer_name_behind = layer_name_list[i_t]
                    layer_name_tail = layer_name_dict[layer_name_behind]
                    i_t += 1
                layer_dict_unique[layer_name]["layer_name_tail"] = layer_name_tail
            else:
                layer_dict_unique[layer_name]["layer_name_tail"] = None

    f.close()
    
    return layer_dict_unique, layer_name_dict

    layer_dict = {}
    input_activation_num = {}
    layer_name_list = []
    abs_path = abs_path = os.path.dirname(os.path.abspath(__file__))
    f = open(abs_path + "/nn_input_noc_nop/" + app_name + ".txt")

    print("network model ----- " + app_name + " -------------")

    lines = f.readlines()
    for line in lines:
        if line.startswith("#"):
            pass
        elif line.startswith("*"):
            line_item = line.split(" ")
            for i in line_item:
                if i == "*" or i == "end":
                    pass
                else:
                    layer_name_list.append(i)
        else:
            line = line.replace("\n","")
            line_item = line.split(" ")
            layer_name = line_item[0]
            if layer_name in layer_name_list:
                H = int(line_item[1])
                M = int(line_item[2])
                P = int(line_item[8])
                Q = int(line_item[9])
                C = int(line_item[3])
                K = int(line_item[7])
                R = int(line_item[4])
                S = int(line_item[4])
                stride = int(line_item[5])
                partition = str(line_item[11])
                layer_dict[layer_name] = {"P":P,"Q":Q,"C":C,"K":K,"R":R,"S":S, "stride":stride, "partition": partition}
                input_activation_num[layer_name] = H * M * C
                print(str(layer_name) + " : " + str(layer_dict[layer_name]))
    f.close()
    return layer_dict, input_activation_num

def getSPPartitonList(num, sp_dim, sp_type, TH = 10):
    list = []
    sp_init = {"P":1, "Q":1, "C":1, "K":1, "R":1, "S":1}
    gen_num = 0
    iter_num = 0
    if sp_type == 1:
        #--- single
        for dim_id in sp_dim:
            dim = dim_list[dim_id]
            sp_dict = copy.deepcopy(sp_init)
            sp_dict[dim] = num
            list.append(sp_dict)
        return list
    elif sp_type == 3:
        # 均匀
        assert(len(sp_dim) == 2)
        num_half = int(pow(num, 0.5))
        #assert(num_half * num_half == num)
        num_list = []
        if num_half * num_half == num:
            num_list.append([num_half, num_half])
        else:
            num1 = num_half - 1
            if num1 == 0:
                num1 = 1
            while num % num1 != 0:
                num1 -= 1
            num2 = int(num / num1)
            num_list.append([num1, num2])
            num_list.append([num2, num1])
        dim1 = dim_list[sp_dim[0]]
        dim2 = dim_list[sp_dim[1]]
        for num_l in num_list:
            sp_dict = copy.deepcopy(sp_init)
            sp_dict[dim1] *= num_l[0]
            sp_dict[dim2] *= num_l[1]
            list.append(sp_dict)
        return list
    elif sp_type == 0:
        #--- no limit
        if len(sp_dim) == 1:
            dim = dim_list[sp_dim[0]]
            sp_dict = copy.deepcopy(sp_init)
            sp_dict[dim] = num
            list.append(sp_dict)
            return list
        else:
            while gen_num < TH:
                ran1 = random.randint(0, len(sp_dim)-1)
                ran2 = random.randint(0, len(sp_dim)-1)
                dim1 = dim_list[sp_dim[ran1]]
                dim2 = dim_list[sp_dim[ran2]]
                [num1, num2, num3] = setPartition_1(num, 2)
                sp_dict = copy.deepcopy(sp_init)
                sp_dict[dim1] *= num1
                sp_dict[dim2] *= num2
                if sp_dict not in list:
                    gen_num += 1
                    list.append(sp_dict)
                
                iter_num +=1
                if iter_num > 1000:
                    break
            return list
    elif sp_type == 2:
        #--- hybrid
        assert(len(sp_dim) >= 2)
        while gen_num < TH:
            id_list = list(range(len(sp_dim)))
            random.shuffle(id_list)
            dim1 = dim_list[sp_dim[id_list[0]]]
            dim2 = dim_list[sp_dim[id_list[1]]]
            [num1, num2, num3] = setPartition_1(num, 2)
            sp_dict = copy.deepcopy(sp_init)
            sp_dict[dim1] *= num1
            sp_dict[dim2] *= num2
            if sp_dict not in list and num1 != 1 and num2 != 1:
                gen_num += 1
                list.append(sp_dict)
            
            iter_num +=1
            if iter_num > 1000:
                break
        return list

def getSpatialParallel(HW_param, chiplet_parallel, PE_parallel, numTH = 20):

    spatial_parallel_init = {"pe":{"P":1, "Q":1, "C":1, "K":1, "R":1, "S":1}, "chiplet":{"P":1, "Q":1, "C":1, "K":1, "R":1, "S":1}, "package":{"P":1, "Q":1, "C":1, "K":1, "R":1, "S":1}}
    for dim in HW_param["intra_PE"]:
        spatial_parallel_init["pe"][dim] = HW_param["intra_PE"][dim]
    
    chiplet_H = HW_param["Chiplet"][0]
    chiplet_W = HW_param["Chiplet"][1]
    chiplet_num = chiplet_H * chiplet_W
    PE_H = HW_param["PE"][0]
    PE_W = HW_param["PE"][1]
    PE_num = PE_H * PE_W

    parallel_select, parallel_type = config_parallel_type(chiplet_parallel,PE_parallel)	
    chiplet_sp_dim = parallel_select["Chiplet"]
    chiplet_sp_type = parallel_type["Chiplet"]
    pe_sp_dim = parallel_select["PE"]
    pe_sp_type = parallel_type["PE"]

    chiplet_list = getSPPartitonList(chiplet_num, chiplet_sp_dim, chiplet_sp_type)
    TH_pe = int(numTH/len(chiplet_list))
    pe_list = getSPPartitonList(PE_num, pe_sp_dim, pe_sp_type, TH_pe)

    spatial_parallel_list = []

    for chiplet in chiplet_list:
        for pe in pe_list:
            spatial_parallel = copy.deepcopy(spatial_parallel_init)
            spatial_parallel["chiplet"] = pe
            spatial_parallel["package"] = chiplet
            spatial_parallel_list.append(spatial_parallel)
    
    return spatial_parallel_list

def randomTest_NoC(mid_result_record_file, nn_name, iterTime, result_dir, save_all_records, record_dir, GaType, HW_param, memory_param, layer_dict, spatial_parallel_list, temporal_level, NoC_param, optimization_objective, layer_name_dict, all_sim_node_num, multi_layer_tag="initial", if_multicast=1, io_die_tag = 1):
    
    best_edp_dict = {}
    best_energy_dict = {}
    best_delay_dict = {}
    best_code_dict = {}
    best_degrade_ratio_dict = {}
    NoC_DR_dict = {}
    L2_to_DRAM_DR_dict = {}
    DRAM_to_L2_DR_dict = {}
    NoP_DR_dict = {}
    best_degrade_ratio_dict_dict = {}
    best_compuation_cycles_dict = {}

    best_sample_dict = {}

    edp_total = 0

    latency_list_dict = {}
    energy_list_dict = {}

    i_act_enough_dict = {}
    task_repeat_nums_dict = {}

    for layer_name in layer_dict:
        # --- 输出文件名 --- #
        if save_all_records == 1:
            record_filename = record_dir + "/" + layer_name + "_" + multi_layer_tag + ".xlsx"
        else:
            record_filename = None
        layer_n = layer_name.split("-")
        workload_name = "{}_{}".format(nn_name, layer_n[0])

        # --- 初始化参数 --- #
        network_param = layer_dict[layer_name]
        i_act_enough = 0
        i_act_enough_dict[layer_name] = i_act_enough
        weight_enough = 0
        par_num = 1

        CodeGen = Encoder(GaType, network_param, temporal_level, debug=0)
        Parser = Decoder(temporal_level, HW_param, network_param)

        best_sample, degrade_ratio_dict, latency_list_d, energy_list_d, task_repeat_nums = \
            randomTest(CodeGen, Parser, iterTime, spatial_parallel_list, memory_param, NoC_param, all_sim_node_num, if_multicast, record_filename, workload_name, i_act_enough, weight_enough, par_num, multi_layer_tag, objective=optimization_objective, architecture=architecture, io_die_tag = io_die_tag)
        best_edp_dict[layer_name] 		= best_sample["edp"]
        best_energy_dict[layer_name] 	= best_sample["energy"]
        best_delay_dict[layer_name] 	= best_sample["delay"]
        best_code_dict[layer_name] 		= best_sample["code"]
        best_degrade_ratio_dict[layer_name] = best_sample["degrade_ratio"]
        NoC_DR_dict[layer_name] = degrade_ratio_dict["NoC"]
        L2_to_DRAM_DR_dict[layer_name] = degrade_ratio_dict["L2_to_DRAM"]
        DRAM_to_L2_DR_dict[layer_name] = degrade_ratio_dict["DRAM_to_L2"]
        best_degrade_ratio_dict_dict[layer_name] = degrade_ratio_dict
        best_compuation_cycles_dict[layer_name] = best_sample["compuation_cycles"]
        edp_total += best_sample["edp"]
        latency_list_dict[layer_name] = latency_list_d
        energy_list_dict[layer_name] = energy_list_d
        best_sample_dict[layer_name] = best_sample
        task_repeat_nums_dict[layer_name] = task_repeat_nums
    
    file_1 = result_dir + "/final_result_record_" + multi_layer_tag + ".txt"
    f = open(file_1,'w')
    print(best_edp_dict, file=f)
    print(best_energy_dict, file=f)
    print(best_delay_dict, file=f)
    print(best_code_dict, file = f)
    print(best_degrade_ratio_dict, file = f)
    print(NoC_DR_dict, file = f)
    print(L2_to_DRAM_DR_dict, file = f)
    print(DRAM_to_L2_DR_dict, file = f)
    print(best_degrade_ratio_dict_dict, file = f)
    print(best_compuation_cycles_dict, file = f)
    print("edp_total: ", edp_total, file = f)
    f.close()

    file_2 = mid_result_record_file
    f_2 = open(file_2, 'w')
    for layer_name, best_sample in best_sample_dict.items():
        print("layer_name\t{}".format(layer_name), file=f_2)
        for index, value in best_sample.items():
            print("---\t{}\t{}".format(index, value), file=f_2)
        print("task_repeat_nums: ", task_repeat_nums_dict[layer_name], file=f_2)
        print("", file=f_2)
    f_2.close()
    return best_sample_dict


if __name__ == '__main__':
    # 目前只支持我们的架构，nnbaton和simba还没添加
    # todo : support nnbaton and simba
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default="mesh_DDR4", help='hardware architecture type (mesh_DDR4, ring_DDR4, cmesh_DDR4)')
    parser.add_argument('--app_name', type=str, default="resnet50", help='NN model name')
    parser.add_argument('--chiplet_num', type=int, default=16, help='chiplet_num')
    parser.add_argument('--chiplet_parallel', type=str, default="All", help='chiplet level spatial parallel type') # K_stable, P_stable, PK_stable, C_stable, KC_stable
    parser.add_argument('--PE_parallel', type=str, default="All", help='PE level spatial parallel type')
    parser.add_argument('--debug_open', type=int, default=0, help='debug mode (will print info)')
    parser.add_argument('--save_all_records', type=int, default=0, help='save all record')
    parser.add_argument('--optimization_objective', type=str, default="edp", help='optimization_objective')
    parser.add_argument('--temporal_level', type=int, default=3, help='temporal level')
    opt = parser.parse_args()

    abs_path = os.path.dirname(os.path.abspath(__file__))
    architecture = opt.architecture
    app_name = opt.app_name
    alg = "random"
    encode_type = 'index'
    dataflow = 'ours'
    chiplet_num = opt.chiplet_num
    chiplet_parallel = opt.chiplet_parallel
    PE_parallel = opt.PE_parallel
    debug_open = opt.debug_open
    save_all_records = opt.save_all_records
    optimization_objective = opt.optimization_objective
    temporal_level = opt.temporal_level

    record_outdir = os.path.join(abs_path, "output_record")
    os.makedirs(record_outdir, exist_ok=True)
    record_outdir = os.path.join(record_outdir, architecture + "_" + app_name)
    os.makedirs(record_outdir, exist_ok=True)
    record_outdir = os.path.join(record_outdir, chiplet_parallel + "_and_" + PE_parallel)
    os.makedirs(record_outdir, exist_ok=True)
    record_outdir = os.path.join(record_outdir, alg + "_" + encode_type + "_" + optimization_objective)
    os.makedirs(record_outdir, exist_ok=True)

    # --- 最终结果记录
    debug_final = 1
    if debug_final:
        file_debug_final = app_name + "_result_record.txt"
        if os.path.exists(file_debug_final) == False:
            f_debug_final = open(file_debug_final, 'w')
        else:
            f_debug_final = open(file_debug_final, 'a')
        print("{:-^120}".format(" SETTING "), file = f_debug_final)
        print("architecture={}".format(architecture), file = f_debug_final)
        print("alg={}".format(alg), file = f_debug_final)
        print("encode_type={}".format(encode_type), file = f_debug_final)
        print("chiplet_num=[{}]".format(chiplet_num), file = f_debug_final)
        print("chiplet_parallel={}, PE_parallel={}".format(chiplet_parallel, PE_parallel), file = f_debug_final)
        print("temporal_level={}".format(temporal_level), file = f_debug_final)
        print("optimization_objective={}".format(optimization_objective), file = f_debug_final)
        print("{:-^120}".format(" RESULT "), file = f_debug_final)
    
    result_outdir = os.path.join(abs_path, "result")
    os.makedirs(result_outdir, exist_ok=True)
    result_outdir = os.path.join(result_outdir, "intraLayer")
    os.makedirs(result_outdir, exist_ok=True)
    result_outdir = os.path.join(result_outdir, architecture + "_" + app_name)
    os.makedirs(result_outdir, exist_ok=True)
    result_outdir = os.path.join(result_outdir, "chiplet_num_"+str(chiplet_num))
    os.makedirs(result_outdir, exist_ok=True)
    result_outdir = os.path.join(result_outdir, alg + "_" + encode_type +"_" +  optimization_objective)
    os.makedirs(result_outdir, exist_ok=True)
    result_outdir = os.path.join(result_outdir, chiplet_parallel + "_and_" + PE_parallel)
    os.makedirs(result_outdir, exist_ok=True)

    mid_result_dir = os.path.join(abs_path, "best_sample_record")
    os.makedirs(mid_result_dir, exist_ok=True)
    mid_result_record = os.path.join(mid_result_dir, "{}_c{}_result.txt".format(app_name, chiplet_num))

    # --- 硬件参数
    HW_param = {"Chiplet":[4,4],"PE":[4,4],"intra_PE":{"C":8,"K":8}}       	
    memory_param = {"OL1":3 ,"OL2":48,"AL1":8,"AL2":48,"WL1":32,"WL2":32}		# mesh mem
    noc_topo = 'Mesh'

    if architecture == "cmesh_DDR4" or architecture == "cmesh_HBM":
        HW_param["maxChiplet"] = [4, 4]
        io_die_tag = 1
    elif architecture == "mesh_DDR4" or architecture == "mesh_HBM":
        HW_param["maxChiplet"] = [4, 4]
        io_die_tag = 0
    elif architecture == "ring_DDR4" or architecture == "ring_HBM":
        HW_param["maxChiplet"] = [4, 2]
        io_die_tag = 0
    else:
        print("Error architecture(), not supported".format(architecture))
        raise NotImplementedError

    NoC_w = HW_param["PE"][1] + 1
    NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
    NoP_w = HW_param["Chiplet"][1] + 1
    NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
    TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}
    
    # --- 生成noc-nop结构图
    NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"], TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = noc_topo)
    if_multicast = 0

    # --- 神经网络参数
    layer_dict, layer_name_dict = getLayerParam(app_name)
    partiton_size_list = {"P":2, "Q":2}
    chiplet_num = HW_param["Chiplet"][0] * HW_param["Chiplet"][1]
    OL2_mem = memory_param["OL2"]*8*1024/act_wgt_width * chiplet_num
    WL2_mem = memory_param["WL2"]*8*1024/act_wgt_width * chiplet_num
    mem_size_dict = {"O":OL2_mem, "W":WL2_mem}
    print("layer_name_dict----------------------------------")
    # print(layer_name_dict)
    layer_num_dict = {}
    for r_name, a_name in layer_name_dict.items():
        if a_name not in layer_num_dict:
            layer_num_dict[a_name] = 0
        layer_num_dict[a_name] += 1
    for name, num in layer_num_dict.items():
        print(name, "\t\t", num)

    # --- 获得空间并行度
    spatial_parallel_list = getSpatialParallel(HW_param, chiplet_parallel, PE_parallel)
    # --- 迭代参数
    num_gen = 10
    num_iter = 1
    iterTime = num_gen * num_iter

    best_sample_dict = randomTest_NoC(mid_result_record, "c{}_{}".format(chiplet_num, app_name), iterTime, result_outdir, save_all_records, record_outdir, encode_type, HW_param, memory_param, layer_dict, spatial_parallel_list, temporal_level, NoC_param, optimization_objective, layer_name_dict, all_sim_node_num, if_multicast=if_multicast, io_die_tag=io_die_tag)

    for layer_name, best_sample in best_sample_dict.items():
        print("{}\t{}".format(layer_name, best_sample["degrade_ratio"] * best_sample["delay"]))
    print("----------------------------------------------------------------")
    print("-------------------------- SIM END -----------------------------")
    print("----------------------------------------------------------------")
