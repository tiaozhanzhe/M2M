import math
import os
from platform import architecture
import sys
import random
import numpy as np
import copy
from enum import Enum
from config import *

# network parameter
# 224 224 64 256 3 3
# 56 56 64 64 1 1
# 56 56 64 64 3 3
# 56 56 64 256 1 1
# 56 56 256 64 1 1
# 56 56 256 128 1 1
# 28 28 128 128 3 3
# 28 28 128 512 1 1
# 28 28 512 128 1 1
# 28 28 512 256 1 1
# 224 224 4 64 3 3 
#P = 224
#Q = 224
#C = 4
#K = 64
#R = 3
#S = 3

#C_new = math.ceil(C/C0)
#K_new = math.ceil(K/K0)
#size = [P,Q,C_new,K_new]

# parallel type
# 0: C-P ; 1: P-P ; 2: PQ-P ; 3: KC-P ; 4: all-type
Par_type = 4

# partition list
#Pset = [] #[P1,P2,P3]
#Qset = [] #[Q1,Q2,Q3]
#Cset = [] #[C1,C2,C3,Cp1,Cp2]
#Kset = [] #[K1,K2,K3,Cp1,Cp2]
#parallel_dim_set = []
#parallel_num_set = []

# encode parameter
dim_list = ['P','Q','C','K','R','S']
dim2id = {'P':0, 'Q':1, 'C':2, 'K':3, 'R':4, 'S':5}
type_list = ["for","parallel-for-1","parallel-for-2"]
architecture = ["pe", "chiplet", "package"]
architecture_dim_num = [6, 4, 4]
#chiplet PE todo
# parallel_select = {"Chiplet":[0,1],"PE":[0,1]}
# parallel_type = {"Chiplet":0,"PE":0} # 2: hybrid, 1 single, 0 no limit, 3: 均匀分布

def config_parallel_type(chiplet_parallel,core_parallel):
    parallel_select = {}; parallel_type = {}
    if chiplet_parallel == "Channel":
        parallel_select["Chiplet"] = [3]
        parallel_type["Chiplet"] = 1
    elif chiplet_parallel == "Pq":
        parallel_select["Chiplet"] = [0,1]
        parallel_type["Chiplet"] = 0
    elif chiplet_parallel == "Hybrid":
        parallel_select["Chiplet"] = [0,3]
        parallel_type["Chiplet"] = 2
    elif chiplet_parallel == "All":
        parallel_select["Chiplet"] = [0,1,3]
        parallel_type["Chiplet"] = 0
    elif chiplet_parallel == "P_K_PK":
        parallel_select["Chiplet"] = [0,3]
        parallel_type["Chiplet"] = 0
    elif chiplet_parallel == "P_stable":
        parallel_select["Chiplet"] = [0]
        parallel_type["Chiplet"] = 1
    elif chiplet_parallel == "K_stable":
        parallel_select["Chiplet"] = [3]
        parallel_type["Chiplet"] = 1
    elif chiplet_parallel == "PK_stable" or chiplet_parallel == "PK_1_stable" or chiplet_parallel ==  "PK_2_stable":
        parallel_select["Chiplet"] = [0,3]
        parallel_type["Chiplet"] = 3
    elif chiplet_parallel == "C_stable":
        parallel_select["Chiplet"] = [2]
        parallel_type["Chiplet"] = 1
    elif chiplet_parallel == "KC_stable":
        parallel_select["Chiplet"] = [2,3]
        parallel_type["Chiplet"] = 3
    elif chiplet_parallel == "KC":
        parallel_select["Chiplet"] = [2,3]
        parallel_type["Chiplet"] = 0
    else:
        print("fatal: chiplet_parallel not defined")
        sys.exit()

    if core_parallel == "Channel":
        parallel_select["PE"] = [3]
        parallel_type["PE"] = 1
    elif core_parallel == "PQ":
        parallel_select["PE"] = [0,1]
        parallel_type["PE"] = 0
    elif core_parallel == "PK":
        parallel_select["PE"] = [0,3]
        parallel_type["PE"] = 2
    elif core_parallel == "All":
        parallel_select["PE"] = [0,1,3]
        parallel_type["PE"] = 0
    elif core_parallel == "P_K_PK":
        parallel_select["PE"] = [0,3]
        parallel_type["PE"] = 0
    elif core_parallel == "KC":
        parallel_select["PE"] = [2,3]
        parallel_type["PE"] = 0
    else:
        print("fatal: core_parallel not defined")
        sys.exit()

    return parallel_select, parallel_type

# fitness parameter
O_correlation = [1,1,0,1,0,0]
A_correlation = [1,1,1,0,0,0]
W_correlation = [0,0,1,1,1,1]
# PE Distribution
# set 16
mem_16 = {"o": 0, "a": 5, "w" : 10, "noc-chiplet": 0}
set_1_e_16 = {}
set_2_e_8 = {}
set_8_e_2 = {}
set_16_e_1 = {}
set_4_e_4 = {}
set_1_e_16[0] = {0:[1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19]}
set_16_e_1[0] = {0:[1],1:[2],2:[3],3:[4],4:[6],5:[7],6:[8],7:[9],8:[11],9:[12],10:[13],11:[14],12:[16],13:[17],14:[18],15:[19]}
set_2_e_8[0] = {0:[1,2,3,4,11,12,13,14],1:[6,7,8,9,16,17,18,19]}
set_2_e_8[1] = {0:[1,2,3,4,6,7,8,9],1:[11,12,13,14,16,17,18,19]}
set_2_e_8[2] = {0:[1,2,6,7,11,12,16,17],1:[3,4,8,9,13,14,18,19]}
set_2_e_8[3] = {0:[1,3,6,8,11,13,16,18],1:[2,4,7,9,12,14,17,19]}
set_8_e_2[0] = {0:[1,6],1:[2,7],2:[3,8],3:[4,9],4:[11,16],5:[12,17],6:[13,18],7:[14,19]}
set_8_e_2[1] = {0:[1,11],1:[2,12],2:[3,13],3:[4,14],4:[6,16],5:[7,17],6:[8,18],7:[9,19]}
set_8_e_2[2] = {0:[1,3],1:[2,4],2:[6,8],3:[7,9],4:[11,13],5:[12,14],6:[16,18],7:[17,19]}
set_8_e_2[3] = {0:[1,2],1:[3,4],2:[6,7],3:[8,9],4:[11,12],5:[13,14],6:[16,17],7:[18,19]}
set_4_e_4[0] = {0:[1,2,3,4],1:[6,7,8,9],2:[11,12,13,14],3:[16,17,18,19]}
set_4_e_4[1] = {0:[1,2,6,7],1:[3,4,8,9],2:[11,12,16,17],3:[13,14,18,19]}
set_4_e_4[3] = {0:[1,6,11,16],1:[2,7,12,17],2:[3,8,13,18],3:[4,9,14,19]}
set_4_e_4[2] = {0:[1,3,11,13],1:[2,4,12,14],2:[6,8,16,18],3:[7,9,17,19]}
set16 = {1:set_1_e_16, 2:set_2_e_8, 4:set_4_e_4, 8:set_8_e_2, 16:set_16_e_1}

# set 4
mem_4 = {"o": 0, "a": 0, "w" : 3, "noc-chiplet": 0}
set_4_e_1 = {}
set_1_e_4 = {}
set_2_e_2 = {}
set_4_e_1[0] = {0:[1],1:[2],2:[4],3:[5]}
set_1_e_4[0] = {0:[1,2,4,5]}
set_2_e_2[0] = {0:[1,2],1:[4,5]}
set_2_e_2[1] = {0:[1,4],1:[2,5]}
set_2_e_2[3] = {0:[1,4],1:[2,5]}
set_2_e_2[2] = {0:[1,2],1:[4,5]}
set4 = {1:set_1_e_4, 2:set_2_e_2, 4:set_4_e_1}

# PE Distribution extend
NoC_node_offset_16 = [20,40,60,80,120,140,160,180,220,240,260,280,320,340,360,380]
NoP2NoCnode_16 = [0,20,40,60,80,0,120,140,160,180,0,220,240,260,280,0,320,340,360,380]
NoC_node_offset_4 = [20,40,80,100]
NoP2NoCnode_4 = [0,20,40,0,80,100]
# A_W_offset = {"o":2, "a":7, "w":12}
A_W_offset = {"o":0, "a":5, "w":10}