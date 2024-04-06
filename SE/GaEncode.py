import math
from operator import index
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from basicParam_noc_nop import *

# 质因数分解
def getZhiyinShu(num, list):
	isZhishu = True
	i = 2
	square = int(math.sqrt(num)) + 1
	while i <= square:
		if num % i == 0:
			list.append(i)
			isZhishu = False
			getZhiyinShu(num / i , list)
			i += 1
			break
		i += 1
	if isZhishu:
		list.append(int(num))

# 将一个数拆成三个整数相乘
def setPartition_1(num, dim):
	par_list = []
	getZhiyinShu(num, par_list)
	par_dim = [1, 1, 1]
	for i in par_list:
		ran = random.randint(1, dim)
		par_dim[ran-1] *= i
	return par_dim

# 将num拆分为dim个整数相乘（大于等于）
def setPartition(num, dim):
	list = []
	num_i = num
	for i in range(0, dim - 1):
		if num_i == 0:
			ran = 1
		else:
			ran = random.randint(1, math.ceil(num_i))
		list.append(ran)
		num_i = num_i / ran

	list.append(math.ceil(num_i))
	random.shuffle(list)
	return list

# 将P_num个并行量进行拆分，随机选两个维度d1、d2，以及对应的并行量p1、p2.
def setParallel(P_num, level, parallel_select, parallel_type, size):
	parallel_t = parallel_type[level]
	list_p = setPartition_1(P_num, 2)
	p1 = list_p[0]
	p2 = list_p[1]
	dim = len(parallel_select[level])

	if parallel_t == 1:
		d2 = d1 = random.randint(0,dim-1)
		p1 = p1 * p2
		p2 = 1
	elif parallel_t == 2:
		list1 = list(range(dim))
		random.shuffle(list1)
		d1 = list1[0]
		d2 = list1[1]
		while (p1 == 1 or p2 == 1):
			list_p = setPartition_1(P_num, 2)
			p1 = list_p[0]
			p2 = list_p[1]
	else:
		d1 = random.randint(0,dim-1)
		d2 = random.randint(0,dim-1)
		if d1 == d2:
			p1 = p1 * p2
			p2 = 1
	d1 = parallel_select[level][d1]
	d2 = parallel_select[level][d2]
	p1 = min(p1, size[dim_list[d1]])
	p2 = min(p2, size[dim_list[d2]])
	return d1, d2, p1, p2

# 只对并行量P_num拆分为p1、p2，不决定其维度
def setParallelNum(P_num):
	index = math.ceil(math.log(P_num, 2))
	p1 = random.randint(1,index-1)
	p2 = index - p1
	assert(p1 >= 1)
	assert(p2 >= 1)
	p1 = pow(2,p1)
	p2 = pow(2,p2)
	return p1, p2

# -- 新编码方案
class GaEncode:
	def __init__(self, GaType, network_param, HW_param, spatial_parallel = None, debug=0, architecture_level=3, debug_file="./random_test_record.txt"):
		self.network_param = network_param
		self.spatial_parallel = spatial_parallel
		self.architecture_level = architecture_level
		self.debug = debug
		self.debug_file = debug_file

		self.temporal_param = None
		#self.getTemporalParam()

		self.GaType = GaType
		self.tile_num_max = 40
		self.loop_tile_dict = {}
		#self.getTileDict()
		self.tile_order_dict = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]

		self.HW_param = HW_param
		self.Chiplets_h = HW_param["Chiplet"][0]
		self.Chiplets_w = HW_param["Chiplet"][1]
		self.Chiplets = self.Chiplets_w * self.Chiplets_h
		self.PEs_h = HW_param["PE"][0]
		self.PEs_w = HW_param["PE"][1]
		self.PEs = self.PEs_w * self.PEs_h
		self.intra_PE = HW_param["intra_PE"]		#["C":4,"K":4]

		self.NoC_node_offset = []
		self.NoP2NoCnode = []
		self.A_W_offset = {}
		self.setNodeID()
	
	def getTileDict(self):
		for dim in self.temporal_param:
			if dim != "R" and dim != "S":
				tile_num = 0
				iter_num = 0
				self.loop_tile_dict[dim] = []
				num = self.temporal_param[dim]
				while (tile_num < self.tile_num_max and iter_num < 500):
					iter_num += 1
					tile_list = setPartition_1(num, 3)
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
					level_name = architecture[level]
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
			Pset = setPartition_1(self.temporal_param["P"], self.architecture_level)
			Qset = setPartition_1(self.temporal_param["Q"], self.architecture_level)
			Cset = setPartition_1(self.temporal_param["C"], self.architecture_level)
			Kset = setPartition_1(self.temporal_param["K"], self.architecture_level)
		else:
			Pset = setPartition(self.temporal_param["P"], self.architecture_level)
			Qset = setPartition(self.temporal_param["Q"], self.architecture_level)
			Cset = setPartition(self.temporal_param["C"], self.architecture_level)
			Kset = setPartition(self.temporal_param["K"], self.architecture_level)
		
		partition_code = Pset[:2] + Qset[:2] + Cset[:2] + Kset[:2]

		return partition_code
	
	def getTempOrderCode(self):
		#---维度排序（时间并行）---
		order_code = []

		for level in range(self.architecture_level):
			for _ in range(architecture_dim_num[level]):
				priority = random.random()
				order_code.append(priority)
		
		return order_code

	def getGaCode_index(self):
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

		Ga_code = partition_code + order_code

		return Ga_code

	def CodeChange_index(self, Ga_code):
		# - spatial for
		parallel_dict = {0:[], 1:[]}
		parallel_dim_dict = {0:[], 1:[]}
		parallel_order = Ga_code[0:2]
		for level in range(self.architecture_level):
			if level > 0:
				level_name = architecture[level]
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
		P_tile_index = Ga_code[0+2]
		P_tile_order = Ga_code[1+2]
		Q_tile_index = Ga_code[2+2]
		Q_tile_order = Ga_code[3+2]
		C_tile_index = Ga_code[4+2]
		C_tile_order = Ga_code[5+2]
		K_tile_index = Ga_code[6+2]
		K_tile_order = Ga_code[7+2]
		order_code = Ga_code[8+2:]
		# -- temporal for code
		for_code = {}
		for_code[0] = [self.loop_tile_dict["P"][P_tile_index][i] for i in self.tile_order_dict[P_tile_order]]
		for_code[1] = [self.loop_tile_dict["Q"][Q_tile_index][i] for i in self.tile_order_dict[Q_tile_order]]
		for_code[2] = [self.loop_tile_dict["C"][C_tile_index][i] for i in self.tile_order_dict[C_tile_order]]
		for_code[3] = [self.loop_tile_dict["K"][K_tile_index][i] for i in self.tile_order_dict[K_tile_order]]

		partition_code_c = for_code[0] + for_code[1] + for_code[2] + for_code[3]

		# -- loop order
		loop_order = {}
		loop_order[0] = order_code[0:6]
		loop_order[1] = order_code[6:10]
		loop_order[2] = order_code[10:]

		loop_order_sorted_index = {}
		m_sorted = sorted(enumerate(loop_order[0]), key=lambda x:x[1])
		loop_order_sorted_index[0] = [m[0] for m in m_sorted]
		m_sorted = sorted(enumerate(loop_order[1]), key=lambda x:x[1])
		loop_order_sorted_index[1] = [m[0] for m in m_sorted]
		m_sorted = sorted(enumerate(loop_order[2]), key=lambda x:x[1])
		loop_order_sorted_index[2] = [m[0] for m in m_sorted]

		loop_order_c = loop_order_sorted_index[0] + loop_order_sorted_index[1] + loop_order_sorted_index[2]

		code_c = parallel_code_c + partition_code_c + loop_order_c

		if self.debug == 1:
			print("Debug in CodeChange :")
			print("--- code_change : ", code_c)

		return code_c

	def getGaCode_num(self):

		pe_parallel_order, chiplet_parallel_order = random.randint(0,1), random.randint(0,1)
		parallel_order_code = [pe_parallel_order, chiplet_parallel_order]


		partition_code = self.getTempPartitonCode()
		order_code = self.getTempOrderCode()

		Ga_code = parallel_order_code + partition_code + order_code

		return Ga_code
	
	def CodeChange_num(self, Ga_code):
		# - spatial for
		parallel_dict = {0:[], 1:[]}
		parallel_dim_dict = {0:[], 1:[]}
		for level in range(self.architecture_level):
			if level > 0:
				level_name = architecture[level]
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
		
		parallel_order = Ga_code[0:2]
		for i in range(len(parallel_order)):
			order = parallel_order[i]
			if order == 1:
				# -- reverse
				parallel_dict[i][0], parallel_dict[i][1] = parallel_dict[i][1], parallel_dict[i][0]
				parallel_dim_dict[i][0], parallel_dim_dict[i][1] = parallel_dim_dict[i][1], parallel_dim_dict[i][0]

		parallel_code_c = parallel_dim_dict[0] + parallel_dim_dict[1] + parallel_dict[0] + parallel_dict[1]
		
		# - temporal
		partition_code = Ga_code[0+2:8+2]
		order_code = Ga_code[8+2:]
		# -- temporal for code
		for_code = {}
		for_code[0] = partition_code[0:2]
		for_code[1] = partition_code[2:4]
		for_code[2] = partition_code[4:6]
		for_code[3] = partition_code[6:]
		# print('Ga_code = ', Ga_code)
		P3 = math.ceil(self.temporal_param["P"] / (partition_code[0] * partition_code[1]))
		Q3 = math.ceil(self.temporal_param["Q"] / (partition_code[2] * partition_code[3]))
		C3 = math.ceil(self.temporal_param["C"] / (partition_code[4] * partition_code[5]))
		K3 = math.ceil(self.temporal_param["K"] / (partition_code[6] * partition_code[7]))
		for_code[0].append(P3); for_code[1].append(Q3);
		for_code[2].append(C3); for_code[3].append(K3); 
		
		partition_code_c = for_code[0] + for_code[1] + for_code[2] + for_code[3]

		# -- loop order
		loop_order = {}
		loop_order[0] = order_code[0:6]
		loop_order[1] = order_code[6:10]
		loop_order[2] = order_code[10:]

		loop_order_sorted_index = {}
		m_sorted = sorted(enumerate(loop_order[0]), key=lambda x:x[1])
		loop_order_sorted_index[0] = [m[0] for m in m_sorted]
		m_sorted = sorted(enumerate(loop_order[1]), key=lambda x:x[1])
		loop_order_sorted_index[1] = [m[0] for m in m_sorted]
		m_sorted = sorted(enumerate(loop_order[2]), key=lambda x:x[1])
		loop_order_sorted_index[2] = [m[0] for m in m_sorted]

		loop_order_c = loop_order_sorted_index[0] + loop_order_sorted_index[1] + loop_order_sorted_index[2]

		code_c = parallel_code_c + partition_code_c + loop_order_c

		return code_c

	def getGaCode(self):
		if self.GaType == "num":
			code = self.getGaCode_num()
		elif self.GaType == "index":
			code = self.getGaCode_index()
		else:
			print("Error GaType({}), (num,index) are provided".format(self.GaType))
			exit()
		return code
	
	def setNodeID(self):
		Chiplet_lenth = self.Chiplets_w
		Chiplet_height = self.Chiplets_h
		PE_lenth = self.PEs_w
		PE_height = self.PEs_h
		assert(Chiplet_lenth*Chiplet_height == self.Chiplets)
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
		for i in range(self.Chiplets):
			if i % Chiplet_lenth == 0: 
				self.NoP2NoCnode.append(0)
			num += PE_num
			self.NoC_node_offset.append(num)
			self.NoP2NoCnode.append(num)

			if (i+1) % Chiplet_lenth == 0:
				num += PE_num
		#print(self.A_W_offset)
		#print(self.NoC_node_offset)
		#print(self.NoP2NoCnode)

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

	# 输出基本参数
	def printBasicSet(self):
		print(" ")
		print("basic set :")
		print("P = " + str(self.network_param["P"]))
		print("Q = " + str(self.network_param["Q"]))
		print("C = " + str(self.network_param["C"]))
		print("K = " + str(self.network_param["K"]))
		print("R = " + str(self.network_param["R"]))
		print("S = " + str(self.network_param["S"]))
		print("Chiplets = " + str(self.HW_param["Chiplet"]))
		print("PEs = " + str(self.HW_param["PE"]))
		print("intra PE parallel : ", self.HW_param["intra_PE"])
		print("")

	def printBasicSetFile(self, file1):
		print(" ", file = file1)
		print("basic set :", file = file1)
		print("P = " + str(self.network_param["P"]), file = file1)
		print("Q = " + str(self.network_param["Q"]), file = file1)
		print("C = " + str(self.network_param["C"]), file = file1)
		print("K = " + str(self.network_param["K"]), file = file1)
		print("R = " + str(self.network_param["R"]), file = file1)
		print("S = " + str(self.network_param["S"]), file = file1)
		print("Chiplets = " + str(self.HW_param["Chiplet"]), file = file1)
		print("PEs = " + str(self.HW_param["PE"]), file = file1)
		print("intra PE parallel : ", self.HW_param["intra_PE"], file = file1)
		print("", file = file1)
	# 把GA code进行解析，解析成{[for_type, dim, for_num]}
	def codeParse(self, code):
		parse_dict = {}
		loop_order = code[20:]
		parallel_code = code[4:8]
		parallel_dim = code[0:4]
		for_code = {}
		for_code[0] = code[8:11]
		for_code[1] = code[11:14]
		for_code[2] = code[14:17]
		for_code[3] = code[17:20]
		line_num = 0
		for i in range(3):
			ii = 3 - i
			if i == 0:
				for j in range(6):
					dim = loop_order[j]
					if dim == 4:
						num = self.temporal_param["R"]
					elif dim == 5:
						num = self.temporal_param["S"]
					else:
						num = for_code[dim][i]

					parse_dict[line_num] = [0,dim,num]
					line_num += 1
			else:
				for j in range(4):
					dim = loop_order[i*4+2 + j]
					num = for_code[dim][i]

					parse_dict[line_num] = [0,dim,num]
					line_num += 1

			if i == 0:
				if parallel_code[0] > 1:
					parse_dict[line_num] = [1,parallel_dim[0],parallel_code[0]]
					line_num += 1
				if parallel_code[1] > 1:
					parse_dict[line_num] = [1,parallel_dim[1],parallel_code[1]]
					line_num += 1
			if i == 1:
				if parallel_code[2] > 1:
					parse_dict[line_num] = [2,parallel_dim[2],parallel_code[2]]
					line_num += 1
				if parallel_code[3] > 1:
					parse_dict[line_num] = [2,parallel_dim[3],parallel_code[3]]
					line_num += 1
		
		#parse_dict[line_num] = [0,3,K0]
		#line_num += 1
		#parse_dict[line_num] = [0,2,C0]

		return parse_dict

	# 输出For循环
	def printParseDict(self, dict):
		for i in reversed(range(len(dict))):
			type = dict[i][0]
			dim = dict[i][1]
			num = dict[i][2]
			if num > 1:
				line = type_list[type] + " " + dim_list[dim] + " in [0 : " + str(num) + "):"
				print(line)

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

	def getPEDistribution16(self, para1, para2):
		assert(para1*para2 == 16)
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		act_PE_dict["send"] = {0:[mem_16["a"]]}
		wgt_PE_dict["send"] = {0:[mem_16["w"]]}

		#---act recv节点列表---
		ran = random.randint(0,len(set16[para1])-1)
		if para1 == 4:
			act_PE_dict["recv"] = set16[para1][ran]
			wgt_PE_dict["recv"] = set16[para1][len(set16[4])-ran-1]
		else:
			act_PE_dict["recv"] = set16[para1][ran]
			wgt_PE_dict["recv"] = set16[para2][ran]
		
		act_set_type = "set_"+str(para1)+"e_"+str(para2)
		wgt_set_type = "set_"+str(para2)+"e_"+str(para1)

		return act_PE_dict, wgt_PE_dict, act_set_type, wgt_set_type

	def getPEDistribution4(self, para1, para2):
		assert(para1*para2 == 4)
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		act_PE_dict["send"] = {0:[mem_4["a"]]}
		wgt_PE_dict["send"] = {0:[mem_4["w"]]}

		ran = random.randint(0,len(set4[para1])-1)
		if para1 == 2:
			act_PE_dict["recv"] = set4[para1][ran]
			wgt_PE_dict["recv"] = set4[para1][len(set4[2])-ran-1]
		else:
			act_PE_dict["recv"] = set4[para1][ran]
			wgt_PE_dict["recv"] = set4[para2][ran]
		
		act_set_type = "set_"+str(para1)+"e_"+str(para2)
		wgt_set_type = "set_"+str(para2)+"e_"+str(para1)

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
		rd_out_PE_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.PEs_h, self.PEs_w, [1,1], [1,1], [runtimeCoreNum, 1])
		wr_out_PE_dict_temp = {"send":rd_out_PE_dict_temp["recv"],"recv":{0:[0]}}
		rd_out_Chiplet_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.Chiplets_h, self.Chiplets_w, [1,1], [1,1], [runtimeChipNum, 1])
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

	# 解析parse，得到对应的输出
	def parseChange(self, parse):
		data_flow = []
		data_flow_dim = []
		ol1_ratio = []
		al1_ratio = []
		wl1_ratio = []
		all_param = []
		out_final = []
		if_act_share_PE = {0:0,1:"0"}
		if_wgt_share_PE = {0:0,1:"0"}
		if_act_share_Chiplet = {0:0,1:"0"}
		if_wgt_share_Chiplet = {0:0,1:"0"}
		parallel_dim_list = {0:[1,1,1,1],1:[1,1,1,1]}
		index_PQKC = [1,1,1,1]
		act_correlate = {"chiplet":[], "PE":[]}
		wgt_correlate = {"chiplet":[], "PE":[]}
		parallel_set_dcit = {"chiplet":[], "PE":[]}
		
		for i in range(len(parse)):
			for_type = parse[i][0]
			dim = parse[i][1]
			num = parse[i][2]
			if for_type == 0:
				if dim < 4:
					data_flow.append(dim_list[dim]+str(index_PQKC[dim]))
					index_PQKC[dim] += 1
				else:
					data_flow.append(dim_list[dim])
				data_flow_dim.append(dim)

				if O_correlation[dim] == 1:
					ol1_ratio.append(num)
				else:
					ol1_ratio.append(1)
				
				if A_correlation[dim] == 1:
					al1_ratio.append(num)
				else:
					al1_ratio.append(1)

				if W_correlation[dim] == 1:
					wl1_ratio.append(num)
				else:
					wl1_ratio.append(1)
				all_param.append(num)
			elif for_type == 1:
				parallel_dim_list[0][dim] *= num
				parallel_set_dcit["PE"].append(num)

				if A_correlation[dim] == 0:
					if_act_share_PE[0] = 1

				act_correlate["PE"].append(A_correlation[dim])
				
				if W_correlation[dim] == 0:
					if_wgt_share_PE[0] = 1

				wgt_correlate["PE"].append(W_correlation[dim])
			elif for_type == 2:
				parallel_dim_list[1][dim] *= num
				parallel_set_dcit["chiplet"].append(num)

				if A_correlation[dim] == 0:
					if_act_share_Chiplet[0] = 1

				act_correlate["chiplet"].append(A_correlation[dim])

				if W_correlation[dim] == 0:
					if_wgt_share_Chiplet[0] = 1
					
				wgt_correlate["chiplet"].append(W_correlation[dim])

		out_id = 0		
		for i in reversed(range(len(ol1_ratio))):
			if ol1_ratio[i] == 1 and O_correlation[data_flow_dim[i]] == 0:
				out_id = i
				break
		for i in range(len(ol1_ratio)):
			if i <= out_id:
				out_final.append(0)
			else:
				out_final.append(1)
		data_flow.append("top")
		ol1_ratio.append(1)
		al1_ratio.append(1)
		wl1_ratio.append(1)
		all_param.append(1)
		out_final.append(1)
		act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution(0, self.PEs_h, self.PEs_w, act_correlate["PE"],wgt_correlate["PE"], parallel_set_dcit["PE"])
		act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution(1, self.Chiplets_h, self.Chiplets_w, act_correlate["chiplet"],wgt_correlate["chiplet"], parallel_set_dcit["chiplet"])
		#if self.PEs == 16:
		#	act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution16(act_share_PE[0],wgt_share_PE[0])
		#elif self.PEs == 4:
		#	act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution4(act_share_PE[0],wgt_share_PE[0])
		#if self.Chiplets == 16:
		#	act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution16(act_share_Chiplet[0],wgt_share_Chiplet[0])
		#elif self.Chiplets == 4:
		#	act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution4(act_share_Chiplet[0],wgt_share_Chiplet[0])
		act_PE_dict = self.getPEExtent(act_PE_dict_temp)
		wgt_PE_dict = self.getPEExtent(wgt_PE_dict_temp)
		act_Chiplet_dict = self.getPEExtent(act_Chiplet_dict_temp,1,"o","a")
		wgt_Chiplet_dict = self.getPEExtent(wgt_Chiplet_dict_temp,1,"o","w")

		if self.debug == 2:
			print("")
			print("Debug in parseChange : ")
			print("--- data_flow : ", data_flow)
			print("--- ol1_ratio : ", ol1_ratio)
			print("--- al1_ratio : ", al1_ratio)
			print("--- wl1_ratio : ", wl1_ratio)
			print("--- all_param : ", all_param)
			print("--- out_final : ", out_final)
			print("--- if_act_share_PE : ", if_act_share_PE)
			print("--- if_wgt_share_PE : ", if_wgt_share_PE)
			print("--- if_act_share_Chiplet : ", if_act_share_Chiplet)
			print("--- if_wgt_share_Chiplet : ", if_wgt_share_Chiplet)

			print("--- act_PE_dict : ", act_PE_dict)
			print("--- wgt_PE_dict : ", wgt_PE_dict)
			print("--- act_Chiplet_dict : ", act_Chiplet_dict)
			print("--- wgt_Chiplet_dict : ", wgt_Chiplet_dict)
			print("--- parallel_dim_list : ", parallel_dim_list)


		return data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet, \
			act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict, parallel_dim_list

	def printOut(self, for_list, act_wgt_dict, parallel_dim_list, out_dict):
		print("data_flow = ",for_list[0])
		print("ol1_ratio = ",for_list[1])
		print("al1_ratio = ",for_list[2])
		print("wl1_ratio = ",for_list[3])
		print("all_param = ",for_list[4])
		print("out_final = ",for_list[5])
		print("parallel_dim_list = ",parallel_dim_list)
		print("")
		print("act_PE_dict = ",act_wgt_dict["act_core"])
		print("")
		print("wgt_PE_dict = ",act_wgt_dict["wgt_core"])
		print("")
		print("act_Chiplet_dict = ",act_wgt_dict["act_chiplet"])
		print("")
		print("wgt_Chiplet_dict = ",act_wgt_dict["wgt_chiplet"])
		print("")
		print("rd_out_PE_dict = ",out_dict["rd_core"])
		print(" ")
		print("wr_out_PE_dict = ",out_dict["wr_core"])
		print(" ")
		print("rd_out_Chiplet_dict = ",out_dict["rd_chip"])
		print(" ")
		print("wr_out_Chiplet_dict = ",out_dict["wr_chip"])
		if self.debug == 1:
			f = open(self.debug_file,'a')
			print("if_act_share_PE = ",for_list[6], file = f)
			print("if_wgt_share_PE = ",for_list[7], file = f)
			print("if_act_share_Chiplet = ",for_list[8], file = f)
			print("if_wgt_share_Chiplet = ",for_list[9], file = f)
			f.close()
	
	# act_PE_dict = {0:[],1:[]}，列表内为act复用的PE list， 0、1代表不同的组
	def GaGetChild(self, Ga_code):
		if self.debug == 1:
			self.printBasicSet()
		if self.GaType == "index":
			code = self.CodeChange_index(Ga_code)
		elif self.GaType == "num":
			code = self.CodeChange_num(Ga_code)
	
		parse = self.codeParse(code)

		if self.debug == 1:
			print("parse: ")
			self.printParseDict(parse)

		#---获得PQKC的拆分---
		partition_list = {"P":[],"Q":[],"K":[],"C":[]}
		partition_list["P"] = code[8:11]
		partition_list["Q"] = code[11:14]
		partition_list["C"] = code[14:17]
		partition_list["K"] = code[17:20]

		for_list = {}
		act_wgt_dict = {}
		for_list[0], for_list[1], for_list[2], for_list[3], for_list[4], for_list[5], for_list[6], for_list[7], for_list[8], for_list[9], act_wgt_dict["act_core"], act_wgt_dict["wgt_core"], act_wgt_dict["act_chiplet"], act_wgt_dict["wgt_chiplet"], parallel_dim_list = self.parseChange(parse)
		
		out_dict = {}
		runtimeCoreNum = 1
		runtimeChipNum = 1
		for mm in range(4):
			runtimeCoreNum *= parallel_dim_list[0][mm]
			runtimeChipNum *= parallel_dim_list[1][mm]
		#print("runtimeCoreNum ",runtimeCoreNum)
		#print("runtimeChipNum ",runtimeChipNum)
		out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = self.getOutputDict(runtimeCoreNum, runtimeChipNum)

		if self.debug == 1:
			self.printOut(for_list, act_wgt_dict, parallel_dim_list, out_dict)

		return for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, code

# -- 之前那一套，包含了空间并行度探索的编码方案
class RandomEncode:
	def __init__(self, network_param, HW_param, debug=0, parallel_level=3, debug_file="./random_test_record.txt", chiplet_parallel="All",core_parallel="All", flag = "ours"):
		self.HW_param = HW_param
		self.Chiplets_h = HW_param["Chiplet"][0]
		self.Chiplets_w = HW_param["Chiplet"][1]
		self.Chiplets = self.Chiplets_w * self.Chiplets_h
		self.PEs_h = HW_param["PE"][0]
		self.PEs_w = HW_param["PE"][1]
		self.PEs = self.PEs_w * self.PEs_h
		self.intra_PE = HW_param["intra_PE"]		#["C":4,"K":4]
		self.parallel_level = parallel_level

		self.network_param = network_param
		self.size = {}
		self.getSize()
		self.P = self.size["P"]
		self.Q = self.size["Q"]
		self.C = self.size["C"]
		self.K = self.size["K"]
		self.R = self.size["R"]
		self.S = self.size["S"]

		self.NoC_node_offset = []
		self.NoP2NoCnode = []
		self.A_W_offset = {}
		self.setNodeID()


		self.flag = flag

		self.debug = debug
		self.debug_file = debug_file

		self.chiplet_parallel = chiplet_parallel

		self.parallel_select, self.parallel_type = config_parallel_type(chiplet_parallel, core_parallel)
	
	def getSize(self):
		for i in self.intra_PE:
			num = math.ceil(self.network_param[i] / self.intra_PE[i])
			self.size[i] = num
		for i in self.network_param:
			if i not in self.size:
				self.size[i] = self.network_param[i]

	def setNodeID(self):
		Chiplet_lenth = self.Chiplets_w
		Chiplet_height = self.Chiplets_h
		PE_lenth = self.PEs_w
		PE_height = self.PEs_h
		assert(Chiplet_lenth*Chiplet_height == self.Chiplets)
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
		for i in range(self.Chiplets):
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

	# 获得for拆分的code
	def getPartitionChild(self):
		size_i = []
		parallel_num_set = []
		parallel_dim_set = []
		for i in range(4):
			dim_num = dim_list[i]
			size_i.append(self.size[dim_num])

		#---并行度拆分（空间并行）---
		if Par_type == 4:
			d1, d2, p1, p2 = setParallel(self.PEs, "PE",self.parallel_select, self.parallel_type, self.size)
			parallel_dim_set.append(d1)
			parallel_dim_set.append(d2)
			parallel_num_set.append(p1)
			parallel_num_set.append(p2)
			size_i[d1] = math.ceil(size_i[d1]/p1)
			size_i[d2] = math.ceil(size_i[d2]/p2)
			
			if self.chiplet_parallel == "P_stable":
				d1 = 0
				d2 = 0
				p1 = self.Chiplets
				p2 = 1
			elif self.chiplet_parallel == "K_stable":
				d1 = 3
				d2 = 3
				p1 = self.Chiplets
				p2 = 1
			elif self.chiplet_parallel == "PK_stable":
				d1 = 0
				d2 = 3
				p1 = int(self.Chiplets**0.5)
				p2 = int(self.Chiplets**0.5)
			elif self.chiplet_parallel == "PK_1_stable":
				assert(self.flag == "nnbaton")
				assert(self.Chiplets == 8)
				d1 = 0
				d2 = 3
				p1 = 4
				p2 = 2
			elif self.chiplet_parallel == "PK_2_stable":
				assert(self.flag == "nnbaton")
				assert(self.Chiplets == 8)
				d1 = 0
				d2 = 3
				p1 = 2
				p2 = 4
			elif self.chiplet_parallel == "C_stable":
				d1 = 2
				d2 = 2
				p1 = self.Chiplets
				p2 = 1
			elif self.chiplet_parallel == "KC_stable":
				d1 = 2
				d2 = 3
				p1 = int(self.Chiplets**0.5)
				p2 = int(self.Chiplets**0.5)
			else:
				d1, d2, p1, p2 = setParallel(self.Chiplets,"Chiplet",self.parallel_select, self.parallel_type, self.size)
			parallel_dim_set.append(d1)
			parallel_dim_set.append(d2)
			parallel_num_set.append(p1)
			parallel_num_set.append(p2)
			size_i[d1] = math.ceil(size_i[d1]/p1)
			size_i[d2] = math.ceil(size_i[d2]/p2)

		#---维度拆分（时间并行）---
		ran1 = random.randint(0,0)
		if ran1 == 0:
			Pset = setPartition_1(size_i[0], self.parallel_level)
			Qset = setPartition_1(size_i[1], self.parallel_level)
			Cset = setPartition_1(size_i[2], self.parallel_level)
			Kset = setPartition_1(size_i[3], self.parallel_level)
		else:
			Pset = setPartition(size_i[0], self.parallel_level)
			Qset = setPartition(size_i[1], self.parallel_level)
			Cset = setPartition(size_i[2], self.parallel_level)
			Kset = setPartition(size_i[3], self.parallel_level)
		for i in range(self.parallel_level,3):
			Pset.append(1)
			Qset.append(1)
			Kset.append(1)
			Cset.append(1)
		code = parallel_dim_set + parallel_num_set + Pset + Qset + Cset + Kset # [Kp1,CP1,Kp2,Cp2,C1~C3,K1~K3,P1~P3,Q1~Q3]
		return code

	# 获得for循环顺序的code
	def getOrderCode(self):
		code = []
		for i in range(3):
			if i == 0:
				list1 = list(range(6))
			else:
				list1 = list(range(4))
			random.shuffle(list1)
			code += list1
		return code

	def codeChange(self, code, type):
		code_par = code[0:20]
		code_order = code[20:]

		parallel_code = code_par[0:8]
		P_par = code_par[8:11]
		Q_par = code_par[11:14]
		C_par = code_par[14:17]
		K_par = code_par[17:20]

		loop1 = code_order[0:6]
		loop2 = code_order[6:10]
		loop3 = code_order[10:14]

		if type == "simba":
			K1 = K_par[0]*K_par[1]*K_par[2]
			C1 = C_par[0]*C_par[1]*C_par[2]
			K_par = [K1,1,1]
			C_par = [C1,1,1]
			loop1 = [1,0,2,3,4,5]
			loop2 = [1,0,2,3]
			loop3 = [1,0,2,3]
		elif type == "nnbaton":
			pass
		code = parallel_code + P_par + Q_par + C_par + K_par + loop1 + loop2 + loop3
		return code

	# 获得GA code（for拆分+for_loop顺序）
	def getChild(self):
		code1 = self.getPartitionChild()
		code2 = self.getOrderCode()
		code = code1 + code2
		if self.flag == "simba":
			code = self.codeChange(code, "simba")
		return code
	
	# 把GA code进行解析，解析成{[for_type, dim, for_num]}
	def codeParse(self, code):
		parse_dict = {}
		loop_order = code[20:]
		parallel_code = code[4:8]
		parallel_dim = code[0:4]
		for_code = {}
		for_code[0] = code[8:11]
		for_code[1] = code[11:14]
		for_code[2] = code[14:17]
		for_code[3] = code[17:20]
		line_num = 0
		for i in range(3):
			ii = 3 - i
			if i == 0:
				for j in range(6):
					dim = loop_order[j]
					if dim == 4:
						num = self.R
					elif dim == 5:
						num = self.S
					else:
						num = for_code[dim][i]

					parse_dict[line_num] = [0,dim,num]
					line_num += 1
			else:
				for j in range(4):
					dim = loop_order[i*4+2 + j]
					num = for_code[dim][i]

					parse_dict[line_num] = [0,dim,num]
					line_num += 1

			if i == 0:
				if parallel_code[0] > 1:
					parse_dict[line_num] = [1,parallel_dim[0],parallel_code[0]]
					line_num += 1
				if parallel_code[1] > 1:
					parse_dict[line_num] = [1,parallel_dim[1],parallel_code[1]]
					line_num += 1
			if i == 1:
				if parallel_code[2] > 1:
					parse_dict[line_num] = [2,parallel_dim[2],parallel_code[2]]
					line_num += 1
				if parallel_code[3] > 1:
					parse_dict[line_num] = [2,parallel_dim[3],parallel_code[3]]
					line_num += 1
		
		#parse_dict[line_num] = [0,3,K0]
		#line_num += 1
		#parse_dict[line_num] = [0,2,C0]

		return parse_dict

	# 输出基本参数
	def printBasicSet(self):
		print(" ")
		print("basic set :")
		print("P = " + str(self.network_param["P"]))
		print("Q = " + str(self.network_param["Q"]))
		print("C = " + str(self.network_param["C"]))
		print("K = " + str(self.network_param["K"]))
		print("R = " + str(self.network_param["R"]))
		print("S = " + str(self.network_param["S"]))
		print("Chiplets = " + str(self.HW_param["Chiplet"]))
		print("PEs = " + str(self.HW_param["PE"]))
		print("intra PE parallel : ", self.HW_param["intra_PE"])
		print("")

	def printBasicSetFile(self, file1):
		print(" ", file = file1)
		print("basic set :", file = file1)
		print("P = " + str(self.network_param["P"]), file = file1)
		print("Q = " + str(self.network_param["Q"]), file = file1)
		print("C = " + str(self.network_param["C"]), file = file1)
		print("K = " + str(self.network_param["K"]), file = file1)
		print("R = " + str(self.network_param["R"]), file = file1)
		print("S = " + str(self.network_param["S"]), file = file1)
		print("Chiplets = " + str(self.HW_param["Chiplet"]), file = file1)
		print("PEs = " + str(self.HW_param["PE"]), file = file1)
		print("intra PE parallel : ", self.HW_param["intra_PE"], file = file1)
		print("", file = file1)

	# 输出For循环
	def printParseDict(self, dict):
		for i in reversed(range(len(dict))):
			type = dict[i][0]
			dim = dict[i][1]
			num = dict[i][2]
			if num > 1:
				line = type_list[type] + " " + dim_list[dim] + " in [0 : " + str(num) + "):"
				print(line)

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

	def getPEDistribution16(self, para1, para2):
		assert(para1*para2 == 16)
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		act_PE_dict["send"] = {0:[mem_16["a"]]}
		wgt_PE_dict["send"] = {0:[mem_16["w"]]}

		#---act recv节点列表---
		ran = random.randint(0,len(set16[para1])-1)
		if para1 == 4:
			act_PE_dict["recv"] = set16[para1][ran]
			wgt_PE_dict["recv"] = set16[para1][len(set16[4])-ran-1]
		else:
			act_PE_dict["recv"] = set16[para1][ran]
			wgt_PE_dict["recv"] = set16[para2][ran]
		
		act_set_type = "set_"+str(para1)+"e_"+str(para2)
		wgt_set_type = "set_"+str(para2)+"e_"+str(para1)

		return act_PE_dict, wgt_PE_dict, act_set_type, wgt_set_type

	def getPEDistribution4(self, para1, para2):
		assert(para1*para2 == 4)
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		act_PE_dict["send"] = {0:[mem_4["a"]]}
		wgt_PE_dict["send"] = {0:[mem_4["w"]]}

		ran = random.randint(0,len(set4[para1])-1)
		if para1 == 2:
			act_PE_dict["recv"] = set4[para1][ran]
			wgt_PE_dict["recv"] = set4[para1][len(set4[2])-ran-1]
		else:
			act_PE_dict["recv"] = set4[para1][ran]
			wgt_PE_dict["recv"] = set4[para2][ran]
		
		act_set_type = "set_"+str(para1)+"e_"+str(para2)
		wgt_set_type = "set_"+str(para2)+"e_"+str(para1)

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
		rd_out_PE_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.PEs_h, self.PEs_w, [1,1], [1,1], [runtimeCoreNum, 1])
		wr_out_PE_dict_temp = {"send":rd_out_PE_dict_temp["recv"],"recv":{0:[0]}}
		rd_out_Chiplet_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.Chiplets_h, self.Chiplets_w, [1,1], [1,1], [runtimeChipNum, 1])
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

	# 解析parse，得到对应的输出
	def parseChange(self, parse):
		data_flow = []
		data_flow_dim = []
		ol1_ratio = []
		al1_ratio = []
		wl1_ratio = []
		all_param = []
		out_final = []
		if_act_share_PE = {0:0,1:"0"}
		if_wgt_share_PE = {0:0,1:"0"}
		if_act_share_Chiplet = {0:0,1:"0"}
		if_wgt_share_Chiplet = {0:0,1:"0"}
		parallel_dim_list = {0:[1,1,1,1],1:[1,1,1,1]}
		index_PQKC = [1,1,1,1]
		act_correlate = {"chiplet":[], "PE":[]}
		wgt_correlate = {"chiplet":[], "PE":[]}
		parallel_set_dcit = {"chiplet":[], "PE":[]}
		
		for i in range(len(parse)):
			for_type = parse[i][0]
			dim = parse[i][1]
			num = parse[i][2]
			if for_type == 0:
				if dim < 4:
					data_flow.append(dim_list[dim]+str(index_PQKC[dim]))
					index_PQKC[dim] += 1
				else:
					data_flow.append(dim_list[dim])
				data_flow_dim.append(dim)

				if O_correlation[dim] == 1:
					ol1_ratio.append(num)
				else:
					ol1_ratio.append(1)
				
				if A_correlation[dim] == 1:
					al1_ratio.append(num)
				else:
					al1_ratio.append(1)

				if W_correlation[dim] == 1:
					wl1_ratio.append(num)
				else:
					wl1_ratio.append(1)
				all_param.append(num)
			elif for_type == 1:
				parallel_dim_list[0][dim] *= num
				parallel_set_dcit["PE"].append(num)

				if A_correlation[dim] == 0:
					if_act_share_PE[0] = 1

				act_correlate["PE"].append(A_correlation[dim])
				
				if W_correlation[dim] == 0:
					if_wgt_share_PE[0] = 1

				wgt_correlate["PE"].append(W_correlation[dim])
			elif for_type == 2:
				parallel_dim_list[1][dim] *= num
				parallel_set_dcit["chiplet"].append(num)

				if A_correlation[dim] == 0:
					if_act_share_Chiplet[0] = 1

				act_correlate["chiplet"].append(A_correlation[dim])

				if W_correlation[dim] == 0:
					if_wgt_share_Chiplet[0] = 1
					
				wgt_correlate["chiplet"].append(W_correlation[dim])

		out_id = 0		
		for i in reversed(range(len(ol1_ratio))):
			if ol1_ratio[i] == 1 and O_correlation[data_flow_dim[i]] == 0:
				out_id = i
				break
		for i in range(len(ol1_ratio)):
			if i <= out_id:
				out_final.append(0)
			else:
				out_final.append(1)
		data_flow.append("top")
		ol1_ratio.append(1)
		al1_ratio.append(1)
		wl1_ratio.append(1)
		all_param.append(1)
		out_final.append(1)
		act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution(0, self.PEs_h, self.PEs_w, act_correlate["PE"],wgt_correlate["PE"], parallel_set_dcit["PE"])
		act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution(1, self.Chiplets_h, self.Chiplets_w, act_correlate["chiplet"],wgt_correlate["chiplet"], parallel_set_dcit["chiplet"])
		#if self.PEs == 16:
		#	act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution16(act_share_PE[0],wgt_share_PE[0])
		#elif self.PEs == 4:
		#	act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution4(act_share_PE[0],wgt_share_PE[0])
		#if self.Chiplets == 16:
		#	act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution16(act_share_Chiplet[0],wgt_share_Chiplet[0])
		#elif self.Chiplets == 4:
		#	act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution4(act_share_Chiplet[0],wgt_share_Chiplet[0])
		act_PE_dict = self.getPEExtent(act_PE_dict_temp)
		wgt_PE_dict = self.getPEExtent(wgt_PE_dict_temp)
		act_Chiplet_dict = self.getPEExtent(act_Chiplet_dict_temp,1,"o","a")
		wgt_Chiplet_dict = self.getPEExtent(wgt_Chiplet_dict_temp,1,"o","w")

		return data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet, act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict, parallel_dim_list

	def printOut(self, for_list, act_wgt_dict, parallel_dim_list, out_dict):
		print("data_flow = ",for_list[0])
		print("ol1_ratio = ",for_list[1])
		print("al1_ratio = ",for_list[2])
		print("wl1_ratio = ",for_list[3])
		print("all_param = ",for_list[4])
		print("out_final = ",for_list[5])
		print("parallel_dim_list = ",parallel_dim_list)
		print("")
		print("act_PE_dict = ",act_wgt_dict["act_core"])
		print("")
		print("wgt_PE_dict = ",act_wgt_dict["wgt_core"])
		print("")
		print("act_Chiplet_dict = ",act_wgt_dict["act_chiplet"])
		print("")
		print("wgt_Chiplet_dict = ",act_wgt_dict["wgt_chiplet"])
		print("")
		print("rd_out_PE_dict = ",out_dict["rd_core"])
		print(" ")
		print("wr_out_PE_dict = ",out_dict["wr_core"])
		print(" ")
		print("rd_out_Chiplet_dict = ",out_dict["rd_chip"])
		print(" ")
		print("wr_out_Chiplet_dict = ",out_dict["wr_chip"])
		if self.debug == 1:
			f = open(self.debug_file,'a')
			print("if_act_share_PE = ",for_list[6], file = f)
			print("if_wgt_share_PE = ",for_list[7], file = f)
			print("if_act_share_Chiplet = ",for_list[8], file = f)
			print("if_wgt_share_Chiplet = ",for_list[9], file = f)
			f.close()

	# act_PE_dict = {0:[],1:[]}，列表内为act复用的PE list， 0、1代表不同的组
	def GaGetChild(self):
		if self.debug == 1:
			self.printBasicSet()
		code = self.getChild()
		parse = self.codeParse(code)
		if self.debug == 1:
			self.printParseDict(parse)

		#---获得PQKC的拆分---
		partition_list = {"P":[],"Q":[],"K":[],"C":[]}
		partition_list["P"] = code[8:11]
		partition_list["Q"] = code[11:14]
		partition_list["C"] = code[14:17]
		partition_list["K"] = code[17:20]

		for_list = {}
		act_wgt_dict = {}
		for_list[0], for_list[1], for_list[2], for_list[3], for_list[4], for_list[5], for_list[6], for_list[7], for_list[8], for_list[9], act_wgt_dict["act_core"], act_wgt_dict["wgt_core"], act_wgt_dict["act_chiplet"], act_wgt_dict["wgt_chiplet"], parallel_dim_list = self.parseChange(parse)
		
		out_dict = {}
		runtimeCoreNum = 1
		runtimeChipNum = 1
		for mm in range(4):
			runtimeCoreNum *= parallel_dim_list[0][mm]
			runtimeChipNum *= parallel_dim_list[1][mm]
		out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = self.getOutputDict(runtimeCoreNum, runtimeChipNum)

		if self.debug == 1:
			self.printOut(for_list, act_wgt_dict, parallel_dim_list, out_dict)

		return for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, code

	def GaGetChildParse(self, code):
		parse = self.codeParse(code)

		#---获得PQKC的拆分---
		partition_list = {"P":[],"Q":[],"K":[],"C":[]}
		partition_list["P"] = code[8:11]
		partition_list["Q"] = code[11:14]
		partition_list["C"] = code[14:17]
		partition_list["K"] = code[17:20]

		for_list = {}

		act_wgt_dict = {}
		for_list[0], for_list[1], for_list[2], for_list[3], for_list[4], for_list[5], for_list[6], for_list[7], for_list[8], for_list[9], act_wgt_dict["act_core"], act_wgt_dict["wgt_core"], act_wgt_dict["act_chiplet"], act_wgt_dict["wgt_chiplet"], parallel_dim_list = self.parseChange(parse)

		out_dict = {}
		out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = self.getOutputDict()
		
		
		return for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list

if __name__ == '__main__':
	from single_engine_predict_intralayer_iodie import *
	from mesh_hetero import *
	GaType = "index"
	debug=1

	network_param = {"P":128,"Q":128,"C":1024,"K":1024,"R":3,"S":3, "stride":1}
	HW_param = {"Chiplet":[2,2],"PE":[4,4],"intra_PE":{"C":16,"K":16}}       	# from granularity exploration
	spatial_parallel = {"pe":{"P":1, "Q":1, "C":16, "K":16, "R":1, "S":1}, "chiplet":{"P":2, "Q":1, "C":1, "K":8, "R":1, "S":1}, "package":{"P":2, "Q":1, "C":1, "K":2, "R":1, "S":1}}
	GATest = GaEncode(GaType, network_param, HW_param, spatial_parallel, debug)

	code = [1, 1, 2, 0, 3, 2, 1, 5, \
		0.3324464496340356, 0.06212728110904364, 0.47419675476838363, 0.3697726441412743, 0.44944442260650497, 0.7631418462128823, \
		0.9938977572483843, 0.9592447541559118, 0.05711266412372018, 0.9462344134381174, \
		0.30288891916933736, 0.1132904681257626, 0.5477972360137251, 0.6722887737006555]
	#code_c = GATest.CodeChange_index(code)
	for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, code = GATest.GaGetChild(code)
	memory_param = {"OL1":8 ,"OL2":128,"AL1":16,"AL2":256,"WL1":64,"WL2":1024}		# from granularity exploration
	NoC_w = HW_param["PE"][1] + 1
	NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
	NoP_w = HW_param["Chiplet"][1] + 1
	NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
	TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}
	
	# --- 生成noc-nop结构图
	NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = 'Ring')
	if_multicast = 1

	calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, network_param, HW_param, memory_param, NoC_param, if_multicast)

