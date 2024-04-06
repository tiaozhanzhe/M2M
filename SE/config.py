nop_bandwidth 	= 100				# Gb/s  # (simba 100Gb/s 1.8Ghz)
noc_bandwidth 	= 68				# Gb/s 	# (simba 68Gb/s  2Ghz)
act_wgt_width  	= 8 				# bit   # same with NNbaton
psum_width		= 24				# bit 	# same with NNbaton
PE_freq 		= 2					# Ghz	# NNbaton 500Mhz
compuation_ability = 128			# Tops
MAC_NUM			= compuation_ability*1024/PE_freq/2
noc_link_width = noc_bandwidth / PE_freq # bit
neu_per_flit_act_wgt = int ( noc_link_width / act_wgt_width ) 
neu_per_flit_psum = int ( noc_link_width  / psum_width )
flit_per_pkt = 5
nop_link_width = nop_bandwidth / PE_freq # bit
neu_per_flit_act_nop = int ( nop_link_width / act_wgt_width )
neu_per_flit_psum_nop = int ( nop_link_width  / psum_width )

freq_1G = 1 * 1000 * 1000 * 1000 
DRAM_energy_ratio = 8.75    	# PJ/bit  from NNbaton
def SRAM_energy(size):			# PJ/bit  from NNbaton
	# return 0.016452 * size + 0.283548
	return 0.81
DIE2DIE_energy_ratio = 1.17
NOC_energy_ratio = 0
MAC_energy_ratio = 0.024    	# 8bit MAC

area_MAC = 135.1					# um^2 from NNbaton
def SRAM_area(size):
	return 518.4 * size + 5440  	# um^2/KB from NNbaton
area_GRS = 0.38 * 1000 * 1000		# 不确定1个or多个 TODO
def area2SRAM(area):
	return int((area-5440)/518.4 )  # KB

area_noc_router = 19 * 1000			# um^2 from simba
arae_nop_router = 42 * 1000			# um^2 from simba
buffer_noc_router = 27 * 1000		# um^2 from simba

## applications
BATCH_SIZE = 128 					#TODO 目前还没有加上
vgg16_conv1 = {"P":224,"Q":224,"C":3,"K":64,"R":3,"S":3,"stride":1}
vgg16_conv12 =  {"P":16,"Q":16,"C":512,"K":512,"R":3,"S":3,"stride":1} # 原本是14
resnet50_conv1 =  {"P":112,"Q":112,"C":3,"K":64,"R":7,"S":7,"stride":2}
resnet50_res2a_branch2a = {"P":55,"Q":55,"C":64,"K":64,"R":1,"S":1,"stride":1}
resnet50_res2a_branch2b = {"P":55,"Q":55,"C":64,"K":64,"R":3,"S":3,"stride":1}