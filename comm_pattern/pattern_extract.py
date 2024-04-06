import yaml
import sys
import os
import math
import random

# commom parameter
PE_h = 4
PE_w = 5
CHIP_w = 6

class Pairs:
    def __init__(self, src, dst, comm, tag):
        self.src = src
        self.dst = dst
        self.comm_num = comm
        self.tag = tag
        self.level = self.checkLevel(src, dst)
    
    def checkLevel(self, src, dst):
        intra_chip_nodes = PE_h * PE_w
        if (abs(dst[0]-src) >= intra_chip_nodes):
            # nop level
            return 1
        else:
            # noc level
            return 0
    
    def p_print(self, id = None):
        if id == None:
            print("src:{}, dst={}, comm_num={}, tag={}, level={}".format(self.src, self.dst, self.comm_num, self.tag, self.level))
        else:
            print("id:{}, src:{}, dst={}, comm_num={}, tag={}, level={}".format(id, self.src, self.dst, self.comm_num, self.tag, self.level))

def trun2pair(workload="c1_resnet18_layer1"):
    file='./pattern_file/{}/comm_pattern.yaml'.format(workload)
    with open(file, encoding='utf-8') as rstream:
        data = yaml.load(rstream, yaml.SafeLoader)
    
    multi_pairs = {}
    id = 0
    for src in data:
        for _, dst_list in data[src].items():
            dst = dst_list[2:]
            comm = dst_list[0]
            tag = dst_list[1]
            p = Pairs(src, dst, comm, tag)
            multi_pairs[id] = p
            id += 1
    return multi_pairs

def multicastTrans_NoC(pairs_in, id):
    # transfer multicast into unicast
    dst_list = pairs_in.dst
    if (len(dst_list) == 1):
        return {id: pairs_in}, id+1
    comm_num = pairs_in.comm_num
    tag = pairs_in.tag
    src = pairs_in.src
    
    src_x = src % PE_w
    src_y = src // PE_w
    assert(src_x == 0)
    
    route_dict = {}
    for dst in dst_list:
        dst_x = dst % PE_w
        dst_y = dst // PE_w
        if dst_x not in route_dict:
            route_dict[dst_x] = {0:src_y, 1: None, 2:src_y} # 0: upper than src_y ; 1: equal with src_y; 2: lowwer than src_y
        if (dst_y > route_dict[dst_x][0]):
            route_dict[dst_x][0] = dst_y
        elif (dst_y < route_dict[dst_x][2]):
            route_dict[dst_x][2] = dst_y
        elif (dst_y == src_y):
            route_dict[dst_x][1] = dst_y
    
    pairs_out = {}
    for dst_x, dst_y_dict in route_dict.items():
        dst_y_up = dst_y_dict[0]
        dst_y_mid = dst_y_dict[1]
        dst_y_down = dst_y_dict[2]
        dst_up = dst_y_up * PE_w + dst_x
        dst_mid = src_y * PE_w + dst_x
        dst_down = dst_y_down * PE_w + dst_x
        
        if dst_y_up != src_y and dst_y_down != src_y:
            # 上下都有
            p_up = Pairs(src, [dst_up], comm_num, tag)
            p_down = Pairs(dst_mid, [dst_down], comm_num, tag)
            pairs_out[id] = p_up
            pairs_out[id+1] = p_down
            id += 2
        elif dst_y_up != src_y:
            # 仅上
            p_up = Pairs(src, [dst_up], comm_num, tag)
            pairs_out[id] = p_up
            id += 1
        elif dst_y_down != src_y:
            # 仅下
            p_down = Pairs(src, [dst_down], comm_num, tag)
            pairs_out[id] = p_down
            id += 1
        elif dst_y_mid == src_y:
            # 仅中间
            p_mid = Pairs(src, [dst_mid], comm_num, tag)
            pairs_out[id] = p_mid
            id += 1
    
    return pairs_out, id

def multicastTrans_NoP(pairs_in, id, node_num):
    def id2ChipID(id):
        chip_id = id // (PE_w * PE_h)
        return chip_id
    
    # transfer multicast into unicast
    dst_list = pairs_in.dst
    if (len(dst_list) == 1):
        return {id: pairs_in}, id+1
    comm_num = pairs_in.comm_num
    tag = pairs_in.tag
    src = pairs_in.src
    src_chip = id2ChipID(src)
    src_x = src_chip % CHIP_w
    src_y = src_chip // CHIP_w
    assert src_x == 0 or src_x == CHIP_w-1, 'src_x={}, src={}, dst={}'.format(src_x, src, dst_list)
    
    noc_offset = dst_list[0] % (PE_w*PE_h)
    route_dict = {}
    for dst in dst_list:
        dst_chip = id2ChipID(dst)
        dst_x = dst_chip % CHIP_w
        dst_y = dst_chip // CHIP_w
        if dst_x not in route_dict:
            route_dict[dst_x] = {0:src_y, 1: None, 2:src_y} # 0: upper than src_y ; 1: equal with src_y; 2: lowwer than src_y
        if (dst_y > route_dict[dst_x][0]):
            route_dict[dst_x][0] = dst_y
        elif (dst_y < route_dict[dst_x][2]):
            route_dict[dst_x][2] = dst_y
        elif (dst_y == src_y):
            route_dict[dst_x][1] = dst_y
    
    pairs_out = {}
    for dst_x, dst_y_dict in route_dict.items():
        dst_y_up = dst_y_dict[0]
        dst_y_mid = dst_y_dict[1]
        dst_y_down = dst_y_dict[2]
        
        dst_up_chip = dst_y_up * CHIP_w + dst_x
        dst_mid_chip = src_y * CHIP_w + dst_x
        dst_down_chip = dst_y_down * CHIP_w + dst_x
        
        dst_up = dst_up_chip * (PE_w*PE_h) + noc_offset
        dst_mid = dst_mid_chip * (PE_w*PE_h) + noc_offset
        dst_down = dst_down_chip * (PE_w*PE_h) + noc_offset
        
        if dst_y_up != src_y and dst_y_down != src_y:
            # 上下都有
            dst_mid = dst_mid_chip + node_num
            p_up = Pairs(src, [dst_up], comm_num, tag)
            p_down = Pairs(dst_mid, [dst_down], comm_num, tag)
            pairs_out[id] = p_up
            pairs_out[id+1] = p_down
            id += 2
        elif dst_y_up != src_y:
            # 仅上
            p_up = Pairs(src, [dst_up], comm_num, tag)
            pairs_out[id] = p_up
            id += 1
        elif dst_y_down != src_y:
            # 仅下
            p_down = Pairs(src, [dst_down], comm_num, tag)
            pairs_out[id] = p_down
            id += 1
        elif dst_y_mid == src_y:
            # 仅中间
            p_mid = Pairs(src, [dst_mid], comm_num, tag)
            pairs_out[id] = p_mid
            id += 1
    
    return pairs_out, id

def pair2unicast(multi_pairs_in):
    multi_pairs_out = {}
    id = 0
    for _, pair in multi_pairs_in.items():
        dst_list = pair.dst
        src = pair.src
        comm_num = pair.comm_num
        tag = pair.tag
        for dst in dst_list:
            p = Pairs(src, [dst], comm_num, tag)
            multi_pairs_out[id] = p
            id += 1
    return multi_pairs_out
               
def routeTransfer(multi_pairs_in, node_num):
    id = 0
    multi_pairs_out = {}
    for _, pairs in multi_pairs_in.items():
        if pairs.level == 0:
            pairs_out , id = multicastTrans_NoC(pairs, id)
        else:
            pairs_out , id = multicastTrans_NoP(pairs, id, node_num)
        multi_pairs_out.update(pairs_out)

    return multi_pairs_out

def createTaskFile(multi_pairs, node_num, workload, pattern_tag = "multicast"):
    chip_recv = {}
    for i in range(node_num):
        chip_recv[i] = {1001:0, 1002:0, 1003:0}
        
    send_dict = {}
    for id, comm_p in multi_pairs.items():
        src = comm_p.src
        dst = comm_p.dst[0]
        comm_num = comm_p.comm_num
        tag = comm_p.tag
        assert(len(comm_p.dst)==1)
        chip_recv[dst][tag] += comm_num
        
        if src not in send_dict:
            send_dict[src] = []
        ins = "send {} {} {}".format(dst, comm_num, tag)
        send_dict[src].append(ins)
    
    recv_dict = {}
    for src, recv in chip_recv.items():
        recv_list = []
        for tag, num in recv.items():
            if num > 0:
                ins = "wait {} {}".format(num, tag)
                recv_list.append(ins)
        if len(recv_list) > 0:
            recv_dict[src] = recv_list
    
    trace_dir = "./trace_file/{}/{}".format(pattern_tag, workload)
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir, exist_ok=True)
    for i in range(node_num+node_num//20):
        trace_f = '{}/{}.txt'.format(trace_dir, i)
        t_f = open(trace_f, 'w')
        tag = 0
        if i in send_dict:
            send_ins = send_dict[i]
            random.shuffle(send_ins)
            for ins in send_ins:
                print(ins, file = t_f)
            tag = 1
        if i in recv_dict:
            recv_ins = recv_dict[i]
            for ins in recv_ins:
                print(ins, file = t_f)
            tag = 1
        if tag:
            print("finish", file=t_f)
        t_f.close()  

if __name__=="__main__":
    chip_num = int(sys.argv[1])
    workload = sys.argv[2]
    total_chip_num = math.ceil(chip_num / 8) * 12
    node_num = total_chip_num * 20
    
    if workload == 'all':
        for workload_name in os.listdir("./pattern_file/"):
            if workload_name.startswith('c{}'.format(chip_num)):
                print("workload_name: ", workload_name)
                multi_pairs_in = trun2pair(workload_name)
                multi_pairs_out_multicast = routeTransfer(multi_pairs_in, node_num)
                multi_pairs_out_unicast = pair2unicast(multi_pairs_in)
                createTaskFile(multi_pairs_out_multicast, node_num, workload_name)
                createTaskFile(multi_pairs_out_unicast, node_num, workload_name, pattern_tag = "unicast")
                for id, p in multi_pairs_out_multicast.items():
                    p.p_print(id)
    else:
        multi_pairs_in = trun2pair(workload)
        multi_pairs_out_multicast = routeTransfer(multi_pairs_in, node_num)
        multi_pairs_out_unicast = pair2unicast(multi_pairs_in)
        createTaskFile(multi_pairs_out_multicast, node_num, workload)
        createTaskFile(multi_pairs_out_unicast, node_num, workload, pattern_tag = "unicast")
        #for id, p in multi_pairs_out_multicast.items():
        #    p.p_print(id)
