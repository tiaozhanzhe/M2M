# from linecache import lazycache
import os
import math
import copy
import random
from matplotlib import pyplot as plt
import argparse
import numpy as np
import shutil
import datetime
import sys

sys.path.append("..")
from SE import config

PE_Frequency = config.PE_freq * config.freq_1G
Optimization_Objective_index = {"edp": 0, "energy": 2, "latency": 1}

# --- debug param
debug_in_getIdealParam = False
debug_in_layout = True
debug_in_BWR = False
debug_in_BWR_simple = False
debug_in_evaluation_tp_sp = False
debug_in_evoluation_temporal_spatial = False
debug_in_evoluation_temporal_spatial_simple = False
debug_in_record_fitness_iter = False

cur_dir = os.path.dirname(os.path.abspath(__file__))
SE_evaluation_dir = os.path.join(cur_dir, "../SE/")
nn_param_dir = os.path.join(SE_evaluation_dir, "nn_input_noc_nop")
SE_result_dir = os.path.join(SE_evaluation_dir,"result/intraLayer")

result_outdir = os.path.join(cur_dir,"multi_nn_result")
os.makedirs(result_outdir, exist_ok=True)
result_plot = os.path.join(result_outdir, "plot")
os.makedirs(result_plot, exist_ok=True)

def getNNParam(nn_name):
	# 获取神经网络的计算量和参数量信息
	# --- 对神经网络性能进行理论分析
	nn_file_path = os.path.join(nn_param_dir, nn_name + ".txt")
	nn_f = open(nn_file_path)

	print("network model ----- " + nn_name + " -------------")

	layer_computation_num_dict = {}
	layer_param_num_dict = {}
	lines = nn_f.readlines()
	nnParam_dict = {}
	layer_id = 1
	for line in lines:
		if line.startswith("#") or line.startswith("*"):
			pass
		else:
			line = line.replace("\n","")
			line_item = line.split(" ")
			layer_name = line_item[0]
			#layer_id = int(layer_name.replace("layer",""))
			H = int(line_item[1])
			M = int(line_item[2])
			P = int(line_item[8])
			Q = int(line_item[9])
			C = int(line_item[3])
			K = int(line_item[7])
			R = int(line_item[4])
			S = int(line_item[4])
			layer_computation_num = P * Q * K * R * S * C
			layer_param_num = H*M*C + P*Q*K + R*S*C*K
			layer_computation_num_dict[layer_name] = layer_computation_num
			layer_param_num_dict[layer_name] = layer_param_num
			nnParam_dict[layer_id] = layer_computation_num * layer_param_num * layer_param_num / 10000000000000000
			layer_id += 1
	nn_f.close()
	return nnParam_dict

class multi_network_DSE:
	def __init__(self, architecture, chiplet_num, workload_dict, Optimization_Objective, tp_TH, sp_TH, BW_tag = 1, debug_tag = 0, layout_mapping_method = 'balance'):
		self.architecture = architecture
		self.chiplet_num = chiplet_num
		self.cluster_size = 4
		self.cluster_num = math.ceil(chiplet_num / self.cluster_size)
		self.workload_dict = workload_dict
		self.workload_list = []
		self.debug_tag = debug_tag
		self.BW_Reallocate_tag = BW_tag
		self.layout_mapping_method = layout_mapping_method
		self.Optimization_Objective = Optimization_Objective
		
		# --- TH Param ---
		self.space_tp_TH = tp_TH			# max tp num
		self.space_sp_TH = sp_TH			# max sp num
		self.chiplet_partition_TH = 200		# max chiplet partition num per TP Tile
		# self.chiplet_partition_TH = 2		# max chiplet partition num per TP Tile

		self.workload_schedule_TH = 5000	# max workload schedule method num
		# self.workload_schedule_TH = 5	# max workload schedule method num

		# self.workload_schedule_TH = 0	# max workload schedule method num


		self.chiplet_partition_dict = None

		# --- workload fitness variable ---
		self.ideal_param_dict = None
		self.ideal_param_dict_workload = None
		self.workload_fitness_dict = {}
		self.workload_BWNeeded_dict = {}
		self.workload_FlitNeeded_dict = {}
		self.merge_workload_fitness_dict = None
		self.merge_workload_BWNeeded_dict = None
		self.merge_workload_FlitNeeded_dict = None
		self.merge_workload_dict = None

		# --- record variable ---
		self.fitness_best = None
		self.fitness_record = []
		self.schedule_best = None
		self.schedule_record = []
		self.Nchip_partition_best = None
		self.Nchip_par_record = []
		self.tp_sp_space_best = None
		self.tp_sp_space_record = []
		self.sample_num = 0
		self.merge_workload_dict_best = None

		# --- final result ---
		self.final_workload_BWNeeded_dict = None
		self.final_workload_FN_dict = None
		self.final_merge_workload_fitness_dict = None
		self.final_merge_workload_BWNeeded_dict = None

	
	# initialize: 初始化操作
	def initialize(self):
		self.getIdealParam()
		self.setTotalWorkloadFitness()
		self.best_fitness = None
		self.fitness_record = []
		self.sample_record = []
		self.distance_record = []

	# getIdealParam: 获得理论计算的各任务的参数量信息
	def getIdealParam(self):
		self.ideal_param_dict = {}
		self.ideal_param_dict_workload = {}
		if debug_in_getIdealParam:
			print("DEBUG IN getIdealParam()--------------------------")
			print("---ideal_param_dict_workload: ")
		for nn_name, workload in self.workload_dict.items():
			self.ideal_param_dict[nn_name] = copy.deepcopy(getNNParam(nn_name))
			for w_name, [start_id, end_id] in workload.items():
				workload_name = nn_name + w_name
				self.workload_list.append(workload_name)
				param = 0
				for layer_id in range(start_id, end_id+1):
					param += self.ideal_param_dict[nn_name][layer_id]
				self.ideal_param_dict_workload[workload_name] = param

				if debug_in_getIdealParam:
					print("---{}: {}".format(workload_name, param))
		
		if debug_in_getIdealParam:
			print("END------------------------------------------------")
			exit()
	
	# setTotalWorkloadFitness: 获得每个Workload的性能参数
	def setTotalWorkloadFitness(self):
		for workload in self.workload_list:
			file = "{}/SE_result/{}/{}{}.txt".format(cur_dir, self.architecture, workload, "_headTile")
			f_h = open(file)

			file_m = "{}/SE_result/{}/{}{}.txt".format(cur_dir, self.architecture, workload, "_midTile")
			f_m = open(file_m)

			file_t = "{}/SE_result/{}/{}{}.txt".format(cur_dir, self.architecture, workload, "_tailTile")
			f_t = open(file_t)

			file_a = "{}/SE_result/{}/{}{}.txt".format(cur_dir, self.architecture, workload, "_aloneTile")
			f_a = open(file_a)

			lines_h = f_h.readlines()
			lines_m = f_m.readlines()
			lines_t = f_t.readlines()
			lines_a = f_a.readlines()
			self.workload_fitness_dict[workload] = {}
			for line_id in range(len(lines_h)):
				line_h = lines_h[line_id]
				line_m = lines_m[line_id]
				line_t = lines_t[line_id]
				line_a = lines_a[line_id]
				if line_h.startswith("chiplet"):
					line_item_h = line_h.replace("\n","").split("\t")
					line_item_m = line_m.replace("\n","").split("\t")
					line_item_t = line_t.replace("\n","").split("\t")
					line_item_a = line_a.replace("\n","").split("\t")
					chiplet_num = int(line_item_h[1])
					assert(chiplet_num == int(line_item_m[1]))
					assert(chiplet_num == int(line_item_t[1]))
					assert(chiplet_num == int(line_item_a[1]))
					
					edp_h = float(line_item_h[3])
					latency_h = float(line_item_h[5])
					energy_h = float(line_item_h[7])
					edp_m = float(line_item_m[3])
					latency_m = float(line_item_m[5])
					energy_m = float(line_item_m[7])
					edp_t = float(line_item_t[3])
					latency_t = float(line_item_t[5])
					energy_t = float(line_item_t[7])
					edp_a = float(line_item_a[3])
					latency_a = float(line_item_a[5])
					energy_a = float(line_item_a[7])
					self.workload_fitness_dict[workload][chiplet_num] = {}
					self.workload_fitness_dict[workload][chiplet_num]["headTile"] = [edp_h, latency_h, energy_h]
					self.workload_fitness_dict[workload][chiplet_num]["midTile"] = [edp_m, latency_m, energy_m]
					self.workload_fitness_dict[workload][chiplet_num]["tailTile"] = [edp_t, latency_t, energy_t]
					self.workload_fitness_dict[workload][chiplet_num]["aloneTile"] = [edp_a, latency_a, energy_a]
			
			self.workload_BWNeeded_dict[workload] = {}
			BW_dir_h = cur_dir + "/SE_result/" + self.architecture + "/BW_result/" + workload + "_headTile/"
			BW_dir_m = cur_dir + "/SE_result/" + self.architecture + "/BW_result/" + workload + "_midTile/"
			BW_dir_t = cur_dir + "/SE_result/" + self.architecture + "/BW_result/" + workload + "_tailTile/"
			BW_dir_a = cur_dir + "/SE_result/" + self.architecture + "/BW_result/" + workload + "_aloneTile/"
			for i in range(self.chiplet_num):
				chiplet_num = i + 1
				self.workload_BWNeeded_dict[workload][chiplet_num] = {}
				BW_file_h = BW_dir_h + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f_h = open(BW_file_h)
				lines_h = BW_f_h.readlines()
				BW_file_m = BW_dir_m + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f_m = open(BW_file_m)
				lines_m = BW_f_m.readlines()
				BW_file_t = BW_dir_t + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f_t = open(BW_file_t)
				lines_t = BW_f_t.readlines()
				BW_file_a = BW_dir_a + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f_a = open(BW_file_a)
				lines_a = BW_f_a.readlines()

				for line_id in range(len(lines_h)):
					line_h = lines_h[line_id]
					line_m = lines_m[line_id]
					line_t = lines_t[line_id]
					line_a = lines_a[line_id]
					if line_h.startswith("layer_id") or line_h == "":
						pass
					else:
						line_item_h = line_h.replace("\n","").split("\t")
						layer_id_h = int(line_item_h[0])
						latency_h = float(line_item_h[1])
						NoC_NR_h = float(line_item_h[2])
						L2_to_DRAM_NR_h = float(line_item_h[3])
						DRAM_to_L2_NR_h = float(line_item_h[4])
						BW_list_h = [latency_h, NoC_NR_h, L2_to_DRAM_NR_h, DRAM_to_L2_NR_h]

						line_item_m = line_m.replace("\n","").split("\t")
						layer_id_m = int(line_item_m[0])
						latency_m = float(line_item_m[1])
						NoC_NR_m = float(line_item_m[2])
						L2_to_DRAM_NR_m = float(line_item_m[3])
						DRAM_to_L2_NR_m = float(line_item_m[4])
						BW_list_m = [latency_m, NoC_NR_m, L2_to_DRAM_NR_m, DRAM_to_L2_NR_m]

						line_item_t = line_t.replace("\n","").split("\t")
						layer_id_t = int(line_item_t[0])
						latency_t = float(line_item_t[1])
						NoC_NR_t = float(line_item_t[2])
						L2_to_DRAM_NR_t = float(line_item_t[3])
						DRAM_to_L2_NR_t = float(line_item_t[4])
						BW_list_t = [latency_t, NoC_NR_t, L2_to_DRAM_NR_t, DRAM_to_L2_NR_t]

						line_item_a = line_a.replace("\n","").split("\t")
						layer_id_a = int(line_item_a[0])
						latency_a = float(line_item_a[1])
						NoC_NR_a = float(line_item_a[2])
						L2_to_DRAM_NR_a = float(line_item_a[3])
						DRAM_to_L2_NR_a = float(line_item_a[4])
						BW_list_a = [latency_a, NoC_NR_a, L2_to_DRAM_NR_a, DRAM_to_L2_NR_a]

						assert(layer_id_h == layer_id_m)
						assert(layer_id_h == layer_id_a)
						assert(layer_id_h == layer_id_t)

						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id_h] = {}
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id_h]["headTile"] = BW_list_h
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id_h]["midTile"] = BW_list_m
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id_h]["tailTile"] = BW_list_t
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id_h]["aloneTile"] = BW_list_a
			
			self.workload_FlitNeeded_dict[workload] = {}
			FN_dir_h = cur_dir + "/SE_result/" + self.architecture + "/FN_result/" + workload + "_headTile/"
			FN_dir_m = cur_dir + "/SE_result/" + self.architecture + "/FN_result/" + workload + "_midTile/"
			FN_dir_t = cur_dir + "/SE_result/" + self.architecture + "/FN_result/" + workload + "_tailTile/"
			FN_dir_a = cur_dir + "/SE_result/" + self.architecture + "/FN_result/" + workload + "_aloneTile/"
			for i in range(self.chiplet_num):
				chiplet_num = i + 1
				self.workload_FlitNeeded_dict[workload][chiplet_num] = {}
				FN_file_h = FN_dir_h + "chiplet_num_" + str(chiplet_num) + ".txt"
				FN_f_h = open(FN_file_h)
				lines_h = FN_f_h.readlines()
				FN_file_m = FN_dir_m + "chiplet_num_" + str(chiplet_num) + ".txt"
				FN_f_m = open(FN_file_m)
				lines_m = FN_f_m.readlines()
				FN_file_t = FN_dir_t + "chiplet_num_" + str(chiplet_num) + ".txt"
				FN_f_t = open(FN_file_t)
				lines_t = FN_f_t.readlines()
				FN_file_a = FN_dir_a + "chiplet_num_" + str(chiplet_num) + ".txt"
				FN_f_a = open(FN_file_a)
				lines_a = FN_f_a.readlines()

				for line_id in range(len(lines_h)):
					line_h = lines_h[line_id]
					line_m = lines_m[line_id]
					line_t = lines_t[line_id]
					line_a = lines_a[line_id]
					if line_h.startswith("layer_id") or line_h == "":
						pass
					else:
						line_item_h = line_h.replace("\n","").split("\t")
						layer_id_h = int(line_item_h[0])
						iact_DRAM_FN_h = float(line_item_h[1])
						weight_DRAM_FN_h = float(line_item_h[2])
						iact_L2_FN_h = float(line_item_h[3])
						weight_L2_FN_h = float(line_item_h[4])
						oact_rd_FN_h = float(line_item_h[5])
						oact_wr_FN_h = float(line_item_h[6])
						chiplet_spatial_parallel_h = str(line_item_h[7])
						FN_list_h = [iact_DRAM_FN_h, weight_DRAM_FN_h, iact_L2_FN_h, weight_L2_FN_h, oact_rd_FN_h, oact_wr_FN_h, chiplet_spatial_parallel_h]

						line_item_m = line_m.replace("\n","").split("\t")
						layer_id_m = int(line_item_m[0])
						iact_DRAM_FN_m = float(line_item_m[1])
						weight_DRAM_FN_m = float(line_item_m[2])
						iact_L2_FN_m = float(line_item_m[3])
						weight_L2_FN_m = float(line_item_m[4])
						oact_rd_FN_m = float(line_item_m[5])
						oact_wr_FN_m = float(line_item_m[6])
						chiplet_spatial_parallel_m = str(line_item_m[7])
						FN_list_m = [iact_DRAM_FN_m, weight_DRAM_FN_m, iact_L2_FN_m, weight_L2_FN_m, oact_rd_FN_m, oact_wr_FN_m, chiplet_spatial_parallel_m]


						line_item_t = line_t.replace("\n","").split("\t")
						layer_id_t = int(line_item_t[0])
						iact_DRAM_FN_t = float(line_item_t[1])
						weight_DRAM_FN_t = float(line_item_t[2])
						iact_L2_FN_t = float(line_item_t[3])
						weight_L2_FN_t = float(line_item_t[4])
						oact_rd_FN_t = float(line_item_t[5])
						oact_wr_FN_t = float(line_item_t[6])
						chiplet_spatial_parallel_t = str(line_item_t[7])
						FN_list_t = [iact_DRAM_FN_t, weight_DRAM_FN_t, iact_L2_FN_t, weight_L2_FN_t, oact_rd_FN_t, oact_wr_FN_t, chiplet_spatial_parallel_t]

						line_item_a = line_a.replace("\n","").split("\t")
						layer_id_a = int(line_item_a[0])
						iact_DRAM_FN_a = float(line_item_a[1])
						weight_DRAM_FN_a = float(line_item_a[2])
						iact_L2_FN_a = float(line_item_a[3])
						weight_L2_FN_a = float(line_item_a[4])
						oact_rd_FN_a = float(line_item_a[5])
						oact_wr_FN_a = float(line_item_a[6])
						chiplet_spatial_parallel_a = str(line_item_a[7])
						FN_list_a = [iact_DRAM_FN_a, weight_DRAM_FN_a, iact_L2_FN_a, weight_L2_FN_a, oact_rd_FN_a, oact_wr_FN_a, chiplet_spatial_parallel_a]

						assert(layer_id_h == layer_id_m)
						assert(layer_id_h == layer_id_a)
						assert(layer_id_h == layer_id_t)

						self.workload_FlitNeeded_dict[workload][chiplet_num][layer_id_h] = {}
						self.workload_FlitNeeded_dict[workload][chiplet_num][layer_id_h]["headTile"] = FN_list_h
						self.workload_FlitNeeded_dict[workload][chiplet_num][layer_id_h]["midTile"] = FN_list_m
						self.workload_FlitNeeded_dict[workload][chiplet_num][layer_id_h]["tailTile"] = FN_list_t
						self.workload_FlitNeeded_dict[workload][chiplet_num][layer_id_h]["aloneTile"] = FN_list_a
	
	# calIdealParam: 计算每个[tp_id, sp_id]内任务的理论参数量
	def calIdealParam(self, tp_sp_space):
		tp_sp_idealParam = {}
		for tp_id, sp_space  in tp_sp_space.items():
			tp_sp_idealParam[tp_id] = {}
			for sp_id, workload_list in sp_space.items():
				tp_sp_idealParam[tp_id][sp_id] = 0
				for workload in workload_list:
					tp_sp_idealParam[tp_id][sp_id] += self.ideal_param_dict_workload[workload]
		return tp_sp_idealParam

	# mergeWorkload: 对于一个Tile内的相同网络的子任务进行合并
	def mergeWorkload(self, tp_sp_space):
		self.merge_workload_fitness_dict = {}
		self.merge_workload_BWNeeded_dict = {}
		self.merge_workload_dict = {}
		merge_tp_sp_space = {}
		for tp_id, sp_space in tp_sp_space.items():
			merge_tp_sp_space[tp_id] = {}
			for sp_id, w_list in sp_space.items():
				app_name_list = {}
				merge_tp_sp_space[tp_id][sp_id] = []
				for w_name in w_list:
					w_items = w_name.split("w")
					app_name = w_items[0]
					w_id = int(w_items[1])
					if app_name not in app_name_list:
						app_name_list[app_name] = []
					app_name_list[app_name].append(w_id)

				for app_name, app_w_list in app_name_list.items():
					app_w_list.sort()
					app_w_name_list = []
					w_tag_dict = {}
					merge_workload_name = ""

					if app_name not in self.merge_workload_dict:
						self.merge_workload_dict[app_name] = {}

					for w_id in app_w_list:
						if len(app_w_list) == 1:
							tag = "aloneTile"
							start_layer_id = self.workload_dict[app_name]["w" + str(w_id)][0]
							end_layer_id = self.workload_dict[app_name]["w" + str(w_id)][1]
						elif w_id == app_w_list[0]:
							tag = "headTile"
							start_layer_id = self.workload_dict[app_name]["w" + str(w_id)][0]
						elif w_id == app_w_list[-1]:
							tag = "tailTile"
							end_layer_id = self.workload_dict[app_name]["w" + str(w_id)][1]
						else:
							tag = "midTile"
						merge_workload_name += "w" + str(w_id)
						w_name = app_name + "w" + str(w_id)
						app_w_name_list.append(w_name)
						w_tag_dict[w_name] = tag
					
					self.merge_workload_dict[app_name][merge_workload_name] = [start_layer_id, end_layer_id]
					merge_workload_name = app_name + merge_workload_name
					merge_tp_sp_space[tp_id][sp_id].append(merge_workload_name)

					app_workload_fitness_dict = {}
					app_workload_BW_dict = {}
					layer_offset = 0
					for w_name, tag in w_tag_dict.items():
						for chiplet_num, fitness_list in self.workload_fitness_dict[w_name].items():
							if chiplet_num not in app_workload_fitness_dict:
								app_workload_fitness_dict[chiplet_num] = [0,0,0]
								app_workload_BW_dict[chiplet_num] = {}
							app_workload_fitness_dict[chiplet_num][1] += fitness_list[tag][1]
							app_workload_fitness_dict[chiplet_num][2] += fitness_list[tag][2]
							
							for layer_id, BW_list in self.workload_BWNeeded_dict[w_name][chiplet_num].items():
								layer_id_real = layer_id + layer_offset
								assert(layer_id_real not in app_workload_BW_dict[chiplet_num])
								app_workload_BW_dict[chiplet_num][layer_id_real] = BW_list[tag]
						layer_offset = layer_id_real

					self.merge_workload_fitness_dict[merge_workload_name] = copy.deepcopy(app_workload_fitness_dict)
					self.merge_workload_BWNeeded_dict[merge_workload_name] = copy.deepcopy(app_workload_BW_dict)
					for chiplet_num, fitness_list in self.merge_workload_fitness_dict[merge_workload_name].items():
						edp = fitness_list[1] * fitness_list[2] / PE_Frequency
						self.merge_workload_fitness_dict[merge_workload_name][chiplet_num][0] = edp
		# print('self.merge_workload_fitness_dict = ', self.merge_workload_fitness_dict)
		return merge_tp_sp_space
	
	# getFinalBWFN：得到最终的BW与FN参数
	def getFinalBWFN(self):
		self.final_workload_BWNeeded_dict = {}
		self.final_workload_FN_dict = {}
		tp_sp_space = self.tp_sp_space_best
		chiplet_partition = self.Nchip_partition_best

		for tp_id, sp_space in tp_sp_space.items():
			for sp_id, w_name_list in sp_space.items():
				Nchip = chiplet_partition[tp_id][sp_id]

				for w_name in w_name_list:
					self.final_workload_BWNeeded_dict[w_name] = {}
					self.final_workload_FN_dict[w_name] = {}
					w_name_list = w_name.split("w")
					app_name = w_name_list[0]
					w_id_list = w_name_list[1:]
					workload_tag = {}
					for w_id in w_id_list:
						if len(w_id_list) == 1:
							tag = "aloneTile"
						elif w_id == w_id_list[0]:
							tag = "headTile"
						elif w_id == w_id_list[-1]:
							tag = "tailTile"
						else:
							tag = "midTile"
						w_id_name = app_name + "w" + w_id
						workload_tag[w_id_name] = tag

					layer_offset = 0
					for w_id_name, tag in workload_tag.items():
						for layer_id, BW_list in self.workload_BWNeeded_dict[w_id_name][Nchip].items():
								layer_id_real = layer_id + layer_offset
								assert(layer_id_real not in self.final_workload_BWNeeded_dict[w_name])
								self.final_workload_BWNeeded_dict[w_name][layer_id_real] = BW_list[tag]
								self.final_workload_FN_dict[w_name][layer_id_real] = self.workload_FlitNeeded_dict[w_id_name][Nchip][layer_id][tag]
						layer_offset = layer_id_real

		MNN_result_outdir = os.path.join(cur_dir,"multi_nn_result")
		os.makedirs(MNN_result_outdir, exist_ok=True)
		MNN_result_outdir = os.path.join(MNN_result_outdir,"final_result_output")
		if os.path.exists(MNN_result_outdir):
			shutil.rmtree(MNN_result_outdir)
		os.makedirs(MNN_result_outdir, exist_ok=True)
		BW_dir = os.path.join(MNN_result_outdir,"BW_result")
		os.makedirs(BW_dir, exist_ok=True)
		FN_dir = os.path.join(MNN_result_outdir,"FN_result")
		os.makedirs(FN_dir, exist_ok=True)
		
		title_FN = "layer_id\tiact_DRAM_FN\tweight_DRAM_FN\tiact_L2_FN\tweight_L2_FN\toact_rd_FN\toact_wr_FN\tchiplet_spatial_parallel"
		title_BW = "layer_id\tlatency\tNoC_DR\tL2_to_DRAM_DR\tDRAM_to_L2_DR\t"
		for workload_name in self.final_workload_BWNeeded_dict:
			BW_file = BW_dir + "/" + workload_name + ".txt"
			FN_file = FN_dir + "/" + workload_name + ".txt"
			BW_f = open(BW_file, 'w')
			FN_f = open(FN_file, 'w')
			print(title_BW, file = BW_f)
			print(title_FN, file = FN_f)
			for layer_id in self.final_workload_BWNeeded_dict[workload_name]:
				BW_list = self.final_workload_BWNeeded_dict[workload_name][layer_id]
				FN_list = self.final_workload_FN_dict[workload_name][layer_id]

				BW_line = "{}\t{}\t{}\t{}\t{}".format(layer_id, BW_list[0], BW_list[1], BW_list[2], BW_list[3])
				FN_line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(layer_id, FN_list[0], FN_list[1], FN_list[2], FN_list[3], FN_list[4], FN_list[5], FN_list[6])
				
				print(BW_line, file = BW_f)
				print(FN_line, file = FN_f)
			
			BW_f.close()
			FN_f.close()
		
		sp_tp_result_file = MNN_result_outdir + "/sp_tp_out.txt"
		sp_tp_result_f = open(sp_tp_result_file, 'w')
		for tp_id, sp_space in tp_sp_space.items():
			line = "tp" + str(tp_id)
			for sp_id, workload_list in sp_space.items():
				Nchip = chiplet_partition[tp_id][sp_id]
				line += "\t(sp{}:{})".format(sp_id, Nchip)
				for workload in workload_list:
					line += " " + workload
			print(line, file = sp_tp_result_f)
		sp_tp_result_f.close()

	# getChipletPartionDict: 获得chiplet数目的可能拆分
	# chiplet_partition_dict : 
		## { 1:[[36]],
		##   2:[[1,35],[2,34]...],
		##   ...
		##   max_sp_num:[[],[],...] }
	def getChipletPartionDict(self, max_sp_num):
		par_per_sp_TH = 100
		par_num = 0
		unchange_TH = 50
		unchange_times = 0

		chiplet_partition_dict = {}
		for i in range(max_sp_num):
			sp_num = i + 1
			par_num = 0
			unchange_times = 0
			chiplet_partition_dict[sp_num] = []
			while(1):
				chiplet_partition = []
				Nchip_rest = self.chiplet_num - sp_num
				for i in range(sp_num-1):
					Nchip = random.randint(0, Nchip_rest)
					chiplet_partition.append(Nchip+1)
					Nchip_rest -= Nchip
				chiplet_partition.append(Nchip_rest+1)
				chiplet_partition.sort()
				if chiplet_partition not in chiplet_partition_dict[sp_num]:
					chiplet_partition_dict[sp_num].append(chiplet_partition)
					unchange_times = 0
					par_num += 1
				else:
					unchange_times += 1
					
				if unchange_times == unchange_TH or par_num == par_per_sp_TH:
					break
		return chiplet_partition_dict

	# calSP: SP Tile Fitness Computation
	def calSP(self, fitness_dict, id):
		if id == None:
			sp_name = "("
		else:
			sp_name = "sp" + str(id) + "("
		latency_sp = 0
		energy_sp = 0
		for name, fitness in fitness_dict.items():
			sp_name += name + "_"
			latency_sp += fitness[1]
			energy_sp += fitness[2]
		sp_name = sp_name.strip("_")
		sp_name += ")"

		edp_sp = latency_sp * energy_sp / PE_Frequency
		fitness_list = [edp_sp, latency_sp, energy_sp]
		return sp_name, fitness_list
	
	# calTP: TP Tile Fitness Computation
	def calTP(self, fitness_dict, id):
		tp_name = "tp" + str(id) + "("
		latency_tp = 0
		energy_tp = 0
		for name, fitness in fitness_dict.items():
			tp_name += name + "_"
			if fitness[1] > latency_tp:
				latency_tp = fitness[1]
			energy_tp += fitness[2]
		tp_name = tp_name.strip("_")
		tp_name += ")"

		edp_tp = latency_tp * energy_tp / PE_Frequency
		fitness_list = [edp_tp, latency_tp, energy_tp]
		return tp_name, fitness_list

	# getChipletAllocateMethods_random: get chiplet allocate method randomly
	def getChipletAllocateMethods_random(self, sp_idealParam, TH):
		sp_num = len(sp_idealParam)
		method_num = len(self.chiplet_partition_dict[sp_num])

		# id_list: 随机化选取的方案的id
		# --- 随机获取chiplet allocate方案中的min(method_num, TH)个方案
		id_list = list(range(method_num))
		random.shuffle(id_list)

		sp_idealParam_order = sorted(sp_idealParam.items(), key = lambda x: x[1])
		sp_Nchip_list = []

		for i in range(TH):
			if i >= method_num:
				break
			index = id_list[i]
			chip_num_list = self.chiplet_partition_dict[sp_num][index]	# 1. 获取的划分的方案
			
			# 2. 根据SP Tile理论的参数量sp_idealParam_order对比，分配对应的chiplet数目
			sp_Nchip = {}
			for ii in range(sp_num):
				chip_num = chip_num_list[ii]
				sp_id = sp_idealParam_order[ii][0]
				sp_Nchip[sp_id] = chip_num
			assert(sp_Nchip not in sp_Nchip_list)
			sp_Nchip_list.append(sp_Nchip)
		return sp_Nchip_list

	# evaluation_tp_sp: 评估函数，评估任务调度方案
	# ---include: chiplet allocate + bandwidth reallocate + evaluation
	def evaluation_tp_sp(self, tp_sp_space, tp_sp_idealParam):
		
		if debug_in_evaluation_tp_sp:
			print("DEBUG IN evaluation_tp_sp()----------------------")

		tp_sp_fitness = {}
		best_Nchip_partition = {}

		# 评估开始：

		# 1. 每个TP Tile内部探索与评估，以TP Tile的fitness最优为指标迭代
		# ---每个TP Tile内部进行chiplet Allocate的探索
		# ---每个SP Tile内部进行BW Reallocate的探索
		for tp_id, sp_space in tp_sp_space.items():
			# Evaluation in the same TP Tile
			
			sp_idealParam = tp_sp_idealParam[tp_id]	
			if debug_in_evaluation_tp_sp:
				print('tp_id = ', tp_id)
				print('---sp_idealParam = ', sp_idealParam)

			# 1.1 Randomize Chiplet Allocate Methods
			sp_Nchip_list = self.getChipletAllocateMethods_random(sp_idealParam, self.chiplet_partition_TH)

			# 1.2 Chiplet Allocate 方案遍历
			best_tp_fitness = None
			best_bw_reallocate = None
			best_tp_name = None
			best_Nchip = None
			for sp_Nchip in sp_Nchip_list:
				# 1.2.1 Get Workload Fitness
				sp_workload_fitness = {}
				for sp_id in sp_Nchip:
					sp_workload_fitness[sp_id] = {}
					Nchip = sp_Nchip[sp_id]
					for workload in sp_space[sp_id]:
						fitness = self.merge_workload_fitness_dict[workload][Nchip]
						sp_workload_fitness[sp_id][workload] = fitness
				if debug_in_evaluation_tp_sp:
					print('sp_Nchip = ', sp_Nchip)
					print('sp_workload_fitness = ', sp_workload_fitness)
				
				# 1.2.3 Cal Fitness
				sp_fitness_dict = {}
				for sp_id, workload_fitness in sp_workload_fitness.items():
					sp_name, fitness = self.calSP(workload_fitness, sp_id)
					sp_fitness_dict[sp_name] = fitness
				tp_name, tp_fitness = self.calTP(sp_fitness_dict, tp_id)

				# 1.2.4 迭代比较
				Objective_id = Optimization_Objective_index[self.Optimization_Objective]
				if best_tp_fitness == None or tp_fitness[Objective_id] < best_tp_fitness[Objective_id]:
					best_tp_fitness = copy.deepcopy(tp_fitness)
					best_tp_name = tp_name
					best_Nchip = copy.deepcopy(sp_Nchip)
					final_merge_workload_fitness_dict = self.merge_workload_fitness_dict

				
				self.sample_num += 1

			# 1.3 最优方案记录，得到TP Tile内的评估结果	
			tp_sp_fitness[best_tp_name] = best_tp_fitness
			best_Nchip_partition[tp_id] = copy.deepcopy(best_Nchip)
			if debug_in_evaluation_tp_sp:
				print('best_tp_fitness = ', best_tp_fitness)
				print('best_Nchip_partition[', tp_id, '] = ', best_Nchip)

		# 2. calSP, 对多个TP Tile进行合并评估
		schedule_name, total_fitness = self.calSP(tp_sp_fitness, None)

		if debug_in_evaluation_tp_sp:
			print('best_Nchip_partition = ', best_Nchip_partition)
			print('fitness = ', total_fitness)
		
		#exit()
		return schedule_name, total_fitness, best_Nchip_partition

	# decodeCode: workload_code进行解码，解码成每个[tp_id, sp_id]内包含哪些workload的形式
	def decodeCode(self, code):
		sp_max_dict = {}
		for w_name, sp_tp_id in code.items():
			sp_id = sp_tp_id[1]
			tp_id = sp_tp_id[0]
			if tp_id not in sp_max_dict:
				sp_max_dict[tp_id] = 0
			if sp_id > sp_max_dict[tp_id]:
				sp_max_dict[tp_id] = sp_id
		
		tp_sp_space = {}
		for tp_id in range(len(sp_max_dict)):
			tp_sp_space[tp_id] = {}
			sp_max = sp_max_dict[tp_id]
			for sp_id in range(sp_max+1):
				tp_sp_space[tp_id][sp_id] = []

		for w_name, sp_tp_id in code.items():
			sp_id = sp_tp_id[1]
			tp_id = sp_tp_id[0]
			#if tp_id not in tp_sp_space:
			#	tp_sp_space[tp_id] = {}
			#if sp_id not in tp_sp_space[tp_id]:
			#	tp_sp_space[tp_id][sp_id] = []
			tp_sp_space[tp_id][sp_id].append(w_name)
		return tp_sp_space


	# getWorkloadScheduleCodeList_random: workload的调度，随机获得workload的tp_sp_id映射
	# --- workload_code_list: 任务调度方案列表
	# --- sp_num_max: chiplet拆分的最多块数
	def getWorkloadScheduleCodeList_random(self, tp_TH, sp_TH, TH):
		workload_code_list = []

		app_num = len(self.workload_dict)
		sp_num_max = min(app_num, sp_TH)

		app_id_dict = {}
		for app_id, app_name in enumerate(self.workload_dict):
			app_id_dict[app_name] = app_id

		max_iter = 100
		# max_iter = 1000
		iter_num = 0
		while iter_num < max_iter:
			iter_num += 1
			sp_min = 0
			tp_max = tp_TH - 1
			sp_max = sp_num_max-1
			tp_min = 0

			workload_code = {}

			# tp_set & sp_set_dcit : 为了消除无任务的Tile，便于做到方案的唯一性
			tp_set = [0 for _ in range(int(tp_TH))]
			sp_set_dict = {}

			# 1. 获得模型在每个时间点下的sp id，以模型为部署粒度
			# sp_id_dict: key(tp_id), value(sp_list), 获得每个tp point下的网络模型的sp id
			sp_id_dict = {}
			for tp_id in range(tp_TH):
				sp_set_dict[tp_id] = [0 for _ in range(int(sp_num_max))]
				sp_list = []			# sp_list: key(app_id), value(sp_id)
				# --- 随机获取
				for _ in range(app_num):
					sp_id = random.randint(sp_min, sp_max)
					sp_list.append(sp_id)
				# --- 转换整理：按顺序排布、去除空的SP Tile
				sp_list.sort()
				sp_set = set(sp_list)
				sp_change_dict = {}		# sp_change_dict: key(sp_id_pre), value(sp_id_new), 转换sp id，用于去除空的SP Tile
				for index, sp_id in enumerate(sp_set):
					sp_change_dict[sp_id] = index
				for i, sp_id in enumerate(sp_list):
					sp_list[i] = sp_change_dict[sp_id]
				sp_id_dict[tp_id] = sp_list
			
			# 2. 对网络内的任务进行时间上的调度
			for app_name, workload in self.workload_dict.items():
				# 获得网络内每个任务的tp id
				workload_num = len(workload)
				tp_list = []				# tp_list: key(workload_id), value(tp_id)
				for _ in range(workload_num):
					tp_id = random.randint(tp_min, tp_max)
					tp_list.append(tp_id)
				tp_list.sort()
				
				# 整理记录workload位于的Tile id
				# --- workload_code: key(workload_name), value([tp_id, sp_id])
				for w_id, w_name in enumerate(workload):
					workload_name = app_name + w_name
					tp_id = tp_list[w_id]
					sp_id = sp_id_dict[tp_id][app_id_dict[app_name]]
					workload_code[workload_name] = [tp_id, sp_id]
					tp_set[tp_id] = 1
					sp_set_dict[tp_id][sp_id] = 1
			
			# 3. 消除空Tile，对workload的Tile id进行对应的转换
			tp_change_id_list = []
			sp_change_id_dict = {}
			tp_real = 0
			for tp_id, tp_use in enumerate(tp_set):
				tp_change_id_list.append(tp_real)
				tp_real += tp_use
				
				sp_change_id_dict[tp_id] = []
				sp_set = sp_set_dict[tp_id]
				sp_real = 0
				for sp_id in sp_set:
					sp_change_id_dict[tp_id].append(sp_real)
					sp_real += sp_id
			for workload_name, tp_sp_id in workload_code.items():
				tp_id = tp_change_id_list[tp_sp_id[0]]
				sp_id = sp_change_id_dict[tp_sp_id[0]][tp_sp_id[1]]
				workload_code[workload_name][0], workload_code[workload_name][1] = tp_id, sp_id
			
			# 4. 若方案唯一则记录
			if workload_code not in workload_code_list:
				iter_num = 0 
				workload_code_list.append(workload_code)
			if len(workload_code_list) > TH:
				break
		return workload_code_list, sp_num_max

	#########################################################
	# Layout mapping and bandwidth reallocate 
	#########################################################
	def layout_mapping(self, link_allocation = False, bandwidth_allocation = False):
		if debug_in_layout:
			print(str(sys._getframe().f_lineno) + ': self.tp_sp_space_best = ', self.tp_sp_space_best)
			print(str(sys._getframe().f_lineno) + ': self.Nchip_partition_best = ', self.Nchip_partition_best)
			print(str(sys._getframe().f_lineno) + ': self.final_merge_workload_fitness_dict = ', self.final_merge_workload_fitness_dict)
			print(str(sys._getframe().f_lineno) + ': self.final_workload_BWNeeded_dict = ', self.final_workload_BWNeeded_dict)
			print(str(sys._getframe().f_lineno) + ': self.final_workload_FN_dict = ', self.final_workload_FN_dict)
			
		sp_workload_fitness_bw = {} # result of this function
		
		for tp_id, sp_space in self.tp_sp_space_best.items():
			sp_workload_fitness = {}
			sp_workload_fitness_bw[tp_id] = {}

			for sp_id in sp_space:
				sp_workload_fitness[sp_id] = {}
				Nchip = self.Nchip_partition_best[tp_id][sp_id]
				for workload in sp_space[sp_id]:
					if debug_in_layout:
						print(str(sys._getframe().f_lineno) + ': workload = ', workload)
						print(str(sys._getframe().f_lineno) + ': Nchip = ', Nchip)
						print(str(sys._getframe().f_lineno) + ': self.final_merge_workload_fitness_dict[workload]= ', self.final_merge_workload_fitness_dict[workload])
					sp_workload_fitness[sp_id][workload] = self.final_merge_workload_fitness_dict[workload][Nchip]
			if debug_in_layout:
				print(str(sys._getframe().f_lineno) + ': sp_workload_fitness = ', sp_workload_fitness)

			if len(sp_workload_fitness.keys()) == 1:
				for sp_id in sp_workload_fitness.keys():
					sp_workload_fitness_bw[tp_id][sp_id] = {}
					latency = 0
					for workload, fitness in sp_workload_fitness[sp_id].items():
						sp_workload_fitness_bw[tp_id][sp_id][workload] = copy.deepcopy(fitness)
						latency += fitness[1]
						sp_workload_fitness_bw[tp_id][sp_id][workload][1] = latency
						# print(str(sys._getframe().f_lineno) + ': sp_workload_fitness_bw= ', sp_workload_fitness_bw)
						# print(str(sys._getframe().f_lineno) + ': sp_workload_fitness= ', sp_workload_fitness)
						
						# for layer in self.final_workload_FN_dict[workload].keys():
						# 	    dram_to_L2_F_cur += bw_needed_io_die["input"] / (ddr_bandwidth_io_die_input/noc_bandwidth)
						# 		dram_to_L2_F_cur += bw_needed_io_die["weight"] / (ddr_bandwidth_io_die_weight/noc_bandwidth)
						# 		dram_to_L2_F_cur += bw_needed_io_die["output"] / (ddr_bandwidth_io_die_output/noc_bandwidth)
			else:
				'''
				sp_workload_fitness =  
				{
					0: {
						'resnet18': [1793044.2722481424, 514782.6666666666, 3483109258.2400002], 
						'resnet50': [9630241.645624578, 1326631.466666666, 7259168719.872001]
					}, 
					1: {'resnet50same': [15876227.827040443, 1926204.2666666668, 8242234793.984004]}
				}
				sp_Nchip =  {1: 6, 0: 10}
				'''
				#########################################################
				# init
				#########################################################
				delay_min_dict = {}
				compute_cycle_dict = {}
				degrad_ratio_dict = {}
				dram_to_L2_min_dict = {}
				# L2_to_DRAM_min_dict = {}
				total_cycle_dict = {}
				total_cycle_list = []
				eva_nn_chiplet_num_dict = {}
				fitness_dict = {}

				for sp_id, workload_fitness in sp_workload_fitness.items():
					sp_workload_fitness_bw[tp_id][sp_id] = {}
					Nchip = self.Nchip_partition_best[tp_id][sp_id]
					if debug_in_layout:
						print(str(sys._getframe().f_lineno) + ': workload_fitness.keys() = ', workload_fitness.keys())
						print(str(sys._getframe().f_lineno) + ': Nchip = ', Nchip)
					delay_min_dict[sp_id] = {}
					compute_cycle_dict[sp_id] = {}
					degrad_ratio_dict[sp_id] = {}
					# L2_to_DRAM_min_dict[sp_id] = {}
					dram_to_L2_min_dict[sp_id] = {}
					eva_nn_chiplet_num_dict[sp_id] = Nchip
					fitness_dict[sp_id] = 0
					
					for workload, fitness in workload_fitness.items():
						sp_workload_fitness_bw[tp_id][sp_id][workload] = fitness
						workload_BW_list = self.final_merge_workload_BWNeeded_dict[workload][Nchip]
						fitness_dict[sp_id] += fitness[1]

						if debug_in_layout:
							print(str(sys._getframe().f_lineno) + ': workload_BW_list = ', workload_BW_list)
							print(str(sys._getframe().f_lineno) + ': fitness_dict = ', fitness_dict)
						
						for layer_id, BW_list in workload_BW_list.items():
							# latency = BW_list[0]
							# NoC_NR = BW_list[1]
							# L2_to_DRAM_NR = BW_list[2]
							# DRAM_to_L2_NR = BW_list[3]
							delay_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[0]
							degrad_ratio_dict[sp_id][workload + '_' + str(layer_id)] = max(BW_list[1], BW_list[2], BW_list[3])
							compute_cycle_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[0] / max(BW_list[1], BW_list[2], BW_list[3])
							# L2_to_DRAM_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[2]
							dram_to_L2_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[3]

				if debug_in_layout:
					print(str(sys._getframe().f_lineno) + ': fitness_dict = ', fitness_dict)
					print(str(sys._getframe().f_lineno) + ': degrad_ratio_dict = ', degrad_ratio_dict)
					print(str(sys._getframe().f_lineno) + ': delay_min_dict = ', delay_min_dict)
					print(str(sys._getframe().f_lineno) + ': dram_to_L2_min_dict = ', dram_to_L2_min_dict)
					# print(str(sys._getframe().f_lineno) + ': L2_to_DRAM_min_dict = ', L2_to_DRAM_min_dict)
					print(str(sys._getframe().f_lineno) + ': eva_nn_chiplet_num_dict = ', eva_nn_chiplet_num_dict)

					
				if link_allocation : 
					s = sum(fitness_dict.values())
					for sp_id in fitness_dict.keys():
						fitness_dict[sp_id] = fitness_dict[sp_id] / s

					for sp_id in fitness_dict.keys():
						for layer_id in degrad_ratio_dict[sp_id].keys():
							degrad_ratio_dict[sp_id][layer_id] = degrad_ratio_dict[sp_id][layer_id] / (2 * fitness_dict[sp_id])
							delay_min_dict[sp_id][layer_id] = delay_min_dict[sp_id][layer_id] / (2 * fitness_dict[sp_id])

					if debug_in_layout:
						print(str(sys._getframe().f_lineno) + ': fitness_dict = ', fitness_dict)
						print(str(sys._getframe().f_lineno) + ': degrad_ratio_dict = ', degrad_ratio_dict)
						print(str(sys._getframe().f_lineno) + ': delay_min_dict = ', delay_min_dict)
				
				for nn_name in sp_workload_fitness.keys():
					total_cycle_dict[nn_name] = {}
					cycle = 0
					for layer, item in delay_min_dict[nn_name].items():
						cycle += item
						total_cycle_dict[nn_name][layer] = int(cycle)
						total_cycle_list.append(int(cycle))
				# total_cycle = max(total_cycle_dict.values())
				total_cycle_list.sort()
				if debug_in_layout:
					print(str(sys._getframe().f_lineno) + ': total_cycle_dict = ', total_cycle_dict)
					print(str(sys._getframe().f_lineno) + ': total_cycle_list = ', total_cycle_list)
				#########################################################
				# event
				#########################################################
				if bandwidth_allocation:
					event_dict = {}
					old_tick = 0
					index = 0
					while index < len(total_cycle_list):
						nn_name_list = list(total_cycle_dict.keys())
						tick = total_cycle_list[index]
						state = {}
						for nn_name in total_cycle_dict.keys():
							layer_list = list(total_cycle_dict[nn_name].keys())
							now_layer = layer_list[0]
							for i, layer in enumerate(layer_list):
								if total_cycle_dict[nn_name][layer] == tick:
									now_layer = layer
								elif total_cycle_dict[nn_name][layer] < tick and i < len(layer_list) - 1:
									now_layer = layer_list[i + 1]
								else:
									break
							state[nn_name] = now_layer
						event_dict[tick] = state

						if debug_in_layout:
							print(str(sys._getframe().f_lineno) + ': ', tick, state)
						#########################################################
						# layout mapping
						#########################################################
						degrad_ratio_list = []
						network_list = []
						for nn_name in state.keys():
							degrad_ratio_list = degrad_ratio_list + [degrad_ratio_dict[nn_name][state[nn_name]]] * eva_nn_chiplet_num_dict[nn_name]
							network_list = network_list + [nn_name] * eva_nn_chiplet_num_dict[nn_name]
						
						print(str(sys._getframe().f_lineno) + ': degrad_ratio_list =', degrad_ratio_list)
						print(str(sys._getframe().f_lineno) + ': network_list =', network_list)
						print(str(sys._getframe().f_lineno) + ': state =', state)
						#########################################################
						# layout_mapping_method
						#########################################################
						degrad_ratio_list = np.array(degrad_ratio_list)
						network_list = np.array(network_list)

						if self.layout_mapping_method == 'balance':
							degrad_ratio_list = degrad_ratio_list.reshape([self.cluster_size, self.cluster_num]).T
							network_list = network_list.reshape([self.cluster_size, self.cluster_num]).T
						elif self.layout_mapping_method == 'concentrate':
							degrad_ratio_list = degrad_ratio_list.reshape([self.cluster_num, self.cluster_size])
							network_list = network_list.reshape([self.cluster_num, self.cluster_size])
						elif self.layout_mapping_method == 'random':
							np.random.shuffle(network_list)
							degrad_ratio_list = []
							for nn_name in network_list:
								degrad_ratio_list = degrad_ratio_list + [degrad_ratio_dict[nn_name][state[nn_name]]]
							
							degrad_ratio_list = np.array(degrad_ratio_list)
							network_list = np.array(network_list)
							degrad_ratio_list = degrad_ratio_list.reshape([self.cluster_num, self.cluster_size])
							network_list = network_list.reshape([self.cluster_num, self.cluster_size])
						else:
							raise NotImplementedError

						if debug_in_layout:
							print('degrad_ratio_list = ', degrad_ratio_list)
							print('network_list = ', network_list)
						#########################################################
						# bandwidth allocation 
						#########################################################
						
						for nn_name in nn_name_list:
							if debug_in_layout:
								print(str(sys._getframe().f_lineno) + ': nn_name = ', nn_name)

							if degrad_ratio_dict[nn_name][state[nn_name]] > 1:
								flag = True
								min_degrade = float('inf')

								for i in range(self.cluster_num):
									if nn_name in network_list[i]:
										if sum(degrad_ratio_list[i] < 1) >= 1:
											# can be shared bandwidth
											degrade = sum((1 - degrad_ratio_list[i]) * (degrad_ratio_list[i] < 1))
											if degrade < min_degrade:
												min_degrade = degrade
										else:
											# can not be shared bandwidth for this network
											flag = False
											break
									else:
										continue

								if flag:
									#########################################################
									# improve_ratio
									#########################################################
									if degrad_ratio_dict[nn_name][state[nn_name]] - min_degrade < 1:
										improve_ratio = (degrad_ratio_dict[nn_name][state[nn_name]] - 1) / degrad_ratio_dict[nn_name][state[nn_name]]
									else:
										improve_ratio = min_degrade / degrad_ratio_dict[nn_name][state[nn_name]]
									if debug_in_layout:
										print(str(sys._getframe().f_lineno) + ': nn_name = ', nn_name)
										print(str(sys._getframe().f_lineno) + ': improve_ratio = ', improve_ratio)
										print(str(sys._getframe().f_lineno) + ': tick - old_tick = ', tick - old_tick)
										print(str(sys._getframe().f_lineno) + ': int((tick - old_tick) * improve_ratio) = ', int((tick - old_tick) * improve_ratio))
									#########################################################
									# update
									#########################################################
									for layer in total_cycle_dict[nn_name].keys():
										if total_cycle_dict[nn_name][layer] >= tick:
											total_cycle_dict[nn_name][layer] -= int((tick - old_tick) * improve_ratio)

									total_cycle_list = []
									for nn_name in eva_nn_chiplet_num_dict.keys():
										cycle = 0
										for layer in total_cycle_dict[nn_name].keys():
											total_cycle_list.append(total_cycle_dict[nn_name][layer])
									total_cycle_list.sort()

									tick = total_cycle_list[index]
									if debug_in_layout:
										print('--------------')
										print(str(sys._getframe().f_lineno) + ': index = ', index)
										print(str(sys._getframe().f_lineno) + ': tick = ', tick)
										print(str(sys._getframe().f_lineno) + ': new_total_cycle_dict = ', total_cycle_dict)
										print(str(sys._getframe().f_lineno) + ': total_cycle_list = ', total_cycle_list)
									break

						old_tick = tick
						index += 1

				#########################################################
				# output
				#########################################################
				for sp_id, workload_fitness in sp_workload_fitness.items():
					Nchip = self.Nchip_partition_best[tp_id][sp_id]
					for workload, fitness in workload_fitness.items():
						workload_BW_list = self.final_merge_workload_BWNeeded_dict[workload][Nchip]
						layer_id = list(workload_BW_list.keys())[-1]
						latency_sp = total_cycle_dict[sp_id][workload + '_' + str(layer_id)]
						energy_sp = sp_workload_fitness_bw[tp_id][sp_id][workload][2]
						edp_sp = latency_sp * energy_sp / PE_Frequency
						sp_workload_fitness_bw[tp_id][sp_id][workload] = [edp_sp, latency_sp, energy_sp]
		
		tp_sp_fitness = {}
		for tp_id, sp_space in sp_workload_fitness_bw.items():
			sp_fitness_dict = {}
			for sp_id, workload_fitness in sp_space.items():
				sp_name, fitness = self.calSP(workload_fitness, sp_id)
				sp_fitness_dict[sp_name] = fitness
				sp_fitness_dict[sp_name][1] = list(workload_fitness.values())[-1][1]

			tp_name, tp_fitness = self.calTP(sp_fitness_dict, tp_id)
			tp_sp_fitness[tp_name] = tp_fitness
		schedule_name, total_fitness = self.calSP(tp_sp_fitness, None)
		if debug_in_layout:
			print('sp_workload_fitness_bw = ', sp_workload_fitness_bw)
			print('schedule_name = ', schedule_name)
			print('self.fitness_best = ', self.fitness_best)
			print('total_fitness = ', total_fitness)

		return total_fitness, sp_workload_fitness, sp_workload_fitness_bw



	# evoluation_temporal_spatial: 探索函数，主运行函数
	def evoluation_temporal_spatial(self):
		self.initialize()

		# 1. workload schedule：任务调度方案获取 
		# ---目前方法：随机
		workload_code_list, max_sp_num = self.getWorkloadScheduleCodeList_random(self.space_tp_TH, self.space_sp_TH, self.workload_schedule_TH)
		
		self.chiplet_partition_dict = self.getChipletPartionDict(max_sp_num)

		if debug_in_evoluation_temporal_spatial_simple:
			print("DEBUG IN evoluation_temporal_spatial()------------------------------")
			print("---chiplet_partition_dict: ", self.chiplet_partition_dict)
			print("---Start---------------------------------------------")
		
		# 2. 迭代进化：遍历所有的workload调度的方案
		# --- 进行硬件资源分配和评估
		for workload_code in workload_code_list:
			if debug_in_evoluation_temporal_spatial_simple:
				print("-----Start a new workload mapping----------")
				print("-----workload_code = ", workload_code)

			# 2.1: Workload Schedule Method Decode
			# --- Decode and Workload Merge
			tp_sp_space = self.decodeCode(workload_code)
			tp_sp_idealParam = self.calIdealParam(tp_sp_space)
			merge_tp_sp_space = self.mergeWorkload(tp_sp_space)

			if debug_in_evoluation_temporal_spatial_simple:
				print("------tp_sp_space = ", tp_sp_space)
				print("-----merge_tp_sp_space = ", merge_tp_sp_space)
				print("-----merge_workload_dict = ", self.merge_workload_dict)
			
			tp_sp_space = copy.deepcopy(merge_tp_sp_space)

			# 2.2 : Evaluation
			# --- Chiplet Allocate + BW Reallocate + Method Evaluation
			schedule_name, fitness, Nchip_partition = self.evaluation_tp_sp(tp_sp_space, tp_sp_idealParam)

			# 2.3 : 迭代比较
			ob_id = Optimization_Objective_index[self.Optimization_Objective]
			if self.fitness_best == None or fitness[ob_id] < self.fitness_best[ob_id]:
				self.fitness_best = fitness
				self.schedule_best = schedule_name
				self.tp_sp_space_best = tp_sp_space
				self.Nchip_partition_best = Nchip_partition
				self.merge_workload_dict_best = self.merge_workload_dict
				self.final_merge_workload_fitness_dict = self.merge_workload_fitness_dict
				self.final_merge_workload_BWNeeded_dict = self.merge_workload_BWNeeded_dict
			
			if debug_in_record_fitness_iter:
				print("--------------------------------------------------------")
				print("now_schedule_name: ", schedule_name)
				print("now_fitess: ", fitness)
				print("best_schedule_name: ", self.schedule_best)
				print("best_fitess: ", self.fitness_best)
				print("tp_sp_space_best: ", self.tp_sp_space_best)
				print("merge_workload_dict_best: ", self.merge_workload_dict_best)
				print("final_merge_workload_fitness_dict: ", self.final_merge_workload_fitness_dict)
			
			self.fitness_record.append(fitness)
			self.schedule_record.append(schedule_name)
			self.tp_sp_space_record.append(tp_sp_space)

			if debug_in_evoluation_temporal_spatial:
				print("-----best_fitess: ", self.fitness_best)
				print("-----best_schedule_name: ", self.schedule_best)
				print("-----best_Nchip_partition: ", self.Nchip_partition_best)

		

def plot(nn_name_list, architecture):
	id = 1
	row = len(nn_name_list)
	plt.figure("Fitness per chiplet num")
	for nn_name in nn_name_list:
		plt.subplot(row, 1, id)
		id += 1
		file = "{}/SE_result/{}_{}.txt".format(cur_dir, architecture, nn_name)
		f = open(file)
		lines = f.readlines()
		x = []
		y = []
		for line in lines:
			if line.startswith("chiplet"):
				line_items = line.replace("\n","").split("\t")
				chiplet_num = int(line_items[1])
				fitness = float(line_items[3])
				x.append(chiplet_num)
				y.append(fitness)
		if id > row:
			plt.xlabel("Chiplet Num", fontsize = 10)
		plt.ylabel("Fitness", fontsize = 10)
		plt.bar(x, y, width=0.5,color='rosybrown')
		plt.plot(x,y,color='brown')
		plt.tick_params(labelsize=8)
		for i in range(len(x)):
			plt.scatter(x[i],y[i],s=8,color='brown')
			#xy = (x[i], round(y[i]))
			#plt.annotate("%s" % round(y[i]), xy=xy, xytext=(-20, 10), textcoords='offset points')
		plt.title(nn_name, fontsize = 12, color='brown')
	plt.tight_layout(pad=1.1)
	plt.savefig(cur_dir + "/SE_result/fitness_change_per_Nchiplet_line.png", bbox_inches = 'tight')

# loadWorkloadDict: 加载任务列表
def loadWorkloadDict(workload_file, app_name_list):
	f = open(workload_file)
	lines = f.readlines()
	workload_dict = {}
	for line in lines:
		if line != "":
			line_item = line.split(": ")
			app_name = line_item[0]
			if app_name in app_name_list:
				workload_dict[app_name] = {}
				workload_line = line_item[1]
				workload_line = workload_line.replace("{", "")
				workload_line = workload_line.replace("}", "")
				workload_line = workload_line.replace("\n", "")
				workload_items = workload_line.split(";\t")
				for workload in workload_items:
					if workload != "":
						workload_list = workload.split(":")
						workload_name, layer_list = workload_list[0], workload_list[1]
						workload_name = workload_name.replace(app_name, "")
						layer_list = layer_list.replace("[","").replace("]","")
						layer_list = layer_list.split(",")
						start_layer = int(layer_list[0])
						end_layer = int(layer_list[-1])
						workload_dict[app_name][workload_name] = [start_layer, end_layer]

	return workload_dict

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--architecture', type=str, default="cmesh_HBM", help='architecture')
	parser.add_argument('--nn_list', type=str, default="resnet50", help='neural network, using + as split')
	parser.add_argument('--chiplet_num', type=int, default=16, help='NN model name')
	parser.add_argument('--Optimization_Objective', type=str, default="latency", help='Optimization Objective: edp, energy, latency')
	parser.add_argument('--BW_Reallocator_tag', type=int, default=1, help='use BW Reallocator or not')
	parser.add_argument('--layout_mapping_method', type=str, default='balance', help='balance, concentrate or random')
	# parser.add_argument('--link_allocation', action='store_true', help='take link allocation or not')

	parser.add_argument('--tp_TH', type=int, default=10, help='tp max num')
	parser.add_argument('--sp_TH', type=int, default=10, help='sp max num')
	
	opt = parser.parse_args()
	print(opt)
	architecture = opt.architecture
	chiplet_num = opt.chiplet_num
	Optimization_Objective = opt.Optimization_Objective
	nn_list = opt.nn_list
	nn_list.replace("\n", "")
	nn_name_list = nn_list.split("+")

	# 读取任务列表  
	# workload_dict = {"resnet18":{"":[1,17]}, "resnet50":{"w1":[1,21], "w2":[22,50]}, "VGG16":{"":[1,13]}}
	abs_path = os.path.dirname(os.path.abspath(__file__))
	SE_abs_path = os.path.join(abs_path, "../SE")
	result_outdir = os.path.join(abs_path, "SE_result")
	result_outdir = os.path.join(result_outdir, architecture)
	workload_file = os.path.join(result_outdir, "workload_list.txt")
	print('workload_file = ', workload_file)
	print('nn_name_list = ', nn_name_list)
	workload_dict = loadWorkloadDict(workload_file, nn_name_list)
	print('workload_dict = ', workload_dict)

	# DSE
	MNN_Engine = multi_network_DSE(architecture, chiplet_num, workload_dict, Optimization_Objective, tp_TH=opt.tp_TH, sp_TH=opt.sp_TH, BW_tag=opt.BW_Reallocator_tag, layout_mapping_method=opt.layout_mapping_method)
	start_time = datetime.datetime.now()
	MNN_Engine.evoluation_temporal_spatial()
	#########################################################
	# output
	#########################################################
	MNN_Engine.getFinalBWFN()
	# total_fitness, sp_workload_fitness, sp_workload_fitness_bw = MNN_Engine.layout_mapping(link_allocation = True, bandwidth_allocation = True)
	total_fitness, sp_workload_fitness, sp_workload_fitness_bw = MNN_Engine.layout_mapping()
	print('total_fitness = ', total_fitness)
	
	# 控制台输出
	print("Sim END----------------------------------------------------------")
	print("best schedule name : ", MNN_Engine.schedule_best)
	print("best Nchip Partition : ", MNN_Engine.Nchip_partition_best)
	print("best fitness : ", MNN_Engine.fitness_best)
	print("sp_workload_fitness_bw = ", sp_workload_fitness_bw)

	sum_time = {}
	for nn in nn_name_list:
		sum_time[nn] = 0
	for tid in sp_workload_fitness_bw.keys():
		for sid in sp_workload_fitness_bw[tid].keys():
			last = None
			for workload in sp_workload_fitness_bw[tid][sid].keys():
				print(tid, sid, sp_workload_fitness_bw[tid][sid][workload][1])
				if last != None:
					latency = sp_workload_fitness_bw[tid][sid][workload][1] - last
				else:
					latency = sp_workload_fitness_bw[tid][sid][workload][1]
				for nn in nn_name_list:
					if nn in workload:
						print(tid, sid, workload)
						sum_time[nn] += latency

				last = latency
	
	print(sum_time)

	file_path = cur_dir + "/multi_nn_result/explore_result_total.txt"
	if os.path.exists(file_path) == False:
		result_out_file = open(cur_dir + "/multi_nn_result/explore_result_total.txt", 'w')
	else:
		result_out_file = open(cur_dir + "/multi_nn_result/explore_result_total.txt", 'a')
	end_time = datetime.datetime.now()
	sim_time = end_time - start_time

	print("{:-^120}".format(" SIM TIME "), file = result_out_file)
	print("start_time = {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')), file = result_out_file)
	print("end_time = {}".format(end_time.strftime('%Y-%m-%d %H:%M:%S')), file = result_out_file)
	print("sim_time = {}".format(sim_time), file = result_out_file)

	print("{:-^120}".format(" SETTING "), file = result_out_file)
	print("Chiplet Allocate Add", file = result_out_file)
	print("nn_list = {}".format(nn_name_list), file = result_out_file)
	print("workload = {}".format(workload_dict), file = result_out_file)
	print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
	print("architecture = {}".format(architecture), file = result_out_file)
	print("BW_Reallocator_tag = {} (0: without BWR;  1: with BWR)".format(opt.BW_Reallocator_tag), file = result_out_file)
	print("tp_TH = {}".format(opt.tp_TH), file = result_out_file)
	print("sp_TH = {}".format(opt.sp_TH), file = result_out_file)

	print("{:-^120}".format(" RESULT "), file = result_out_file)
	print("total_sample_num = {}".format(MNN_Engine.sample_num), file = result_out_file)
	print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
	print("merge_workload = {}".format(MNN_Engine.merge_workload_dict_best), file = result_out_file)
	print("{:-<100}".format("schedule space result "), file = result_out_file)
	for tp_id, sp_space in MNN_Engine.tp_sp_space_best.items():
		line = "\ttp_id({})\t".format(tp_id)
		for sp_id in sp_space:
			sp_item = "sp_id({}): ".format(sp_id)
			for workload_name in sp_space[sp_id]:
				sp_item += workload_name + "+"
			sp_item = sp_item[:-1]
			sp_item += "; " + str(MNN_Engine.Nchip_partition_best[tp_id][sp_id])
			line += "{:30}\t\t".format(sp_item)
		print(line, file = result_out_file)
	print("{:-<100}".format(""), file = result_out_file)
	print("fitness_result: edp({}), latency({}), energy({})".format(MNN_Engine.fitness_best[0], MNN_Engine.fitness_best[1], MNN_Engine.fitness_best[2]), file = result_out_file)
	print("{:-^120}".format(" END "), file = result_out_file)
	print("", file = result_out_file)
	print("", file = result_out_file)
	print("", file = result_out_file)
	result_out_file.close()

	# 文本输出：迭代样本完整记录
	sample_record = False
	if sample_record:
		result_out_file = open(cur_dir + "/multi_nn_result/explore_result_sample_record.txt", 'w')
		print("{:-^120}".format(" SIM TIME "), file = result_out_file)
		print("start_time = {}".format(start_time), file = result_out_file)
		print("end_time = {}".format(end_time), file = result_out_file)
		print("sim_time = {}".format(sim_time), file = result_out_file)
		print("", file = result_out_file)

		print("{:-^120}".format(" SETTING "), file = result_out_file)
		print("nn_list = {}".format(nn_name_list), file = result_out_file)
		print("workload = {}".format(workload_dict), file = result_out_file)
		print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
		print("Optimization_Objective = {}".format(Optimization_Objective), file = result_out_file)
		print("BW_Reallocator_tag = {} (0: without BWR;  1: with BWR)".format(opt.BW_Reallocator_tag), file = result_out_file)
		print("", file = result_out_file)

		print("{:-^120}".format(" RESULT "), file = result_out_file)
		print("total_sample_num = {}".format(MNN_Engine.sample_num), file = result_out_file)
		print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
		print("{:-<100}".format("schedule space result "), file = result_out_file)
		for tp_id, sp_space in MNN_Engine.tp_sp_space_best.items():
			line = "\ttp_id({})\t".format(tp_id)
			for sp_id in sp_space:
				sp_item = "sp_id({}): ".format(sp_id)
				for workload_name in sp_space[sp_id]:
					sp_item += workload_name + "+"
				sp_item = sp_item[:-1]
				sp_item += "; " + str(MNN_Engine.Nchip_partition_best[tp_id][sp_id])
				line += "{:30}\t\t".format(sp_item)
			print(line, file = result_out_file)
		print("{:-<100}".format(""), file = result_out_file)
		print("fitness_result: edp({}), latency({}), energy({})".format(MNN_Engine.fitness_best[0], MNN_Engine.fitness_best[1], MNN_Engine.fitness_best[2]), file = result_out_file)
		print("", file = result_out_file)

		print("{:-^120}".format(" SAMPLE RECORD "), file = result_out_file)
		print("", file = result_out_file)
		print("{:-^120}".format(" END "), file = result_out_file)

		id = 0
		for sample in MNN_Engine.schedule_record:
			fitness = str(MNN_Engine.fitness_record[id])
			Nchip = str(MNN_Engine.Nchip_par_record[id])
			id += 1
			print("sample={:100}\t, ---Nchip_par:{:60}\t, ---fitness:\t{:15}\t".format(str(sample), Nchip, str(fitness)), file = result_out_file)

	#MNN_Engine.plot()