from config import *
import random

def find_shortest_ring(Ring, src, dst):
    shortest_sub_Ring = []
    shortest_dist = float('inf')
    for sub_Ring in Ring:
        if (src in sub_Ring) and (dst in sub_Ring):
            dist = (sub_Ring.index(dst) - sub_Ring.index(src)) % len(sub_Ring)
            if dist < shortest_dist:
                shortest_dist = dist
                shortest_sub_Ring = sub_Ring

    return shortest_sub_Ring

def append_ring_route(route, sub_Ring, src, dst):
    if sub_Ring.index(dst) > sub_Ring.index(src):
        for i in range(sub_Ring.index(src), sub_Ring.index(dst)):
            route.append((sub_Ring[i], sub_Ring[i + 1]))
        return route
    else:
        for i in range(sub_Ring.index(src), len(sub_Ring) - 1):
            route.append((sub_Ring[i], sub_Ring[i + 1]))

        route.append((sub_Ring[len(sub_Ring) - 1], sub_Ring[0]))

        for i in range(sub_Ring.index(dst)):
            route.append((sub_Ring[i], sub_Ring[i + 1]))
        return route

def add_randam_ring(Ring, NoC_w, NoC_h, NOC_NODE_NUM, NOP_SIZE):
    x1, y1 = 0, 0
    x2, y2 = 0, 0
    while (x1 >= x2) or (y1 >= y2):
        x1 = random.randint(0, NoC_w - 1)
        y1 = random.randint(0, NoC_h - 1)

        x2 = random.randint(0, NoC_w - 1)
        y2 = random.randint(0, NoC_h - 1)

    # print(x1, x2, y1, y2)
    for nop_id in range (NOP_SIZE):
        sub_Ring = []
        if random.randint(0, 1) > 0: 
            for y in range(y1, y2 + 1):
                id = NOC_NODE_NUM * nop_id + y * NoC_w + x1
                sub_Ring.append(id)
            
            for x in range(x1 + 1, x2):
                id = NOC_NODE_NUM * nop_id + y2 * NoC_w + x
                sub_Ring.append(id)

            for y in range(y2, y1 - 1, -1):
                id = NOC_NODE_NUM * nop_id + y * NoC_w + x2
                sub_Ring.append(id)
            
            for x in range(x2 - 1, x1, -1):
                id = NOC_NODE_NUM * nop_id + y1 * NoC_w + x
                sub_Ring.append(id)
        else:
            for x in range(x1, x2 + 1):
                id = NOC_NODE_NUM * nop_id + y1 * NoC_w + x
                sub_Ring.append(id)

            for y in range(y1 + 1, y2):
                id = NOC_NODE_NUM * nop_id + y * NoC_w + x2
                sub_Ring.append(id)

            for x in range(x2, x1 - 1, -1):     
                id = NOC_NODE_NUM * nop_id + y2 * NoC_w + x
                sub_Ring.append(id)

            for y in range(y2 - 1, y1, -1):
                id = NOC_NODE_NUM * nop_id + y * NoC_w + x1
                sub_Ring.append(id)
            
        Ring.append(sub_Ring)
    return Ring   

def gen_RandomRouterless(NoC_w, NoC_h, NOP_SIZE, constraint, ol2_node):
    NOC_NODE_NUM = NoC_w * NoC_h

    while(True):
        Ring = []
        flag = True

        for i in range (constraint):
            Ring = add_randam_ring(Ring, NoC_w, NoC_h, NOC_NODE_NUM, NOP_SIZE)

        for src in range (0, NOC_NODE_NUM):
            for dst in range (0, NOC_NODE_NUM):
                if src != dst:
                    if((src % NoC_w) == ol2_node) or ((dst % NoC_w) == ol2_node):
                        sub_Ring = find_shortest_ring(Ring, src, dst)
                        if sub_Ring == []:
                            flag = False
                            
            if not flag:
                break
        
        if flag:
            break
    return Ring

def construct_noc_nop_RandomRouterless(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio, constraint, ol2_node):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}

    NoC_h = int(NOC_NODE_NUM / NoC_w)

    Ring = gen_RandomRouterless(NoC_w, NoC_h, NOP_SIZE, NoC_w, ol2_node)
    print('Ring = ', Ring)

    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local, src)] = 0
            F[(src, local)] = 0
            bw_scales[(local, src)] = 1
            bw_scales[(src, local)] = 1
            energy_ratio[(local, src)] = NOC_energy_ratio
            energy_ratio[(src, local)] = NOC_energy_ratio

    for sub_Ring in Ring:
        for i in range(len(sub_Ring)):
            F[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] = 0
            energy_ratio[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] = NOC_energy_ratio

            if (sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)]) in bw_scales.keys():
                bw_scales[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] += 1
            else:
                bw_scales[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] = 1

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous Routerless ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)

    for src in range (0, NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0, NOC_NODE_NUM*NOP_SIZE):
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src

            if chip_id(cur_src) != chip_id(dst):
                if (src != 0) or ((noc_id(dst) % NoC_w) != 0):
                    continue
                
                sub_Ring = find_shortest_ring(Ring, src, comm_id(src))
                noc_route_table[(src, dst)] = append_ring_route(noc_route_table[(src, dst)], sub_Ring, src, comm_id(src))
                    
                # go to the nop node
                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src, dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

            if((noc_id(src) % NoC_w) == 0) or ((noc_id(dst) % NoC_w) == 0):
                if cur_src != dst:
                    sub_Ring = find_shortest_ring(Ring, cur_src, dst)
                    noc_route_table[(src, dst)] = append_ring_route(noc_route_table[(src, dst)], sub_Ring, cur_src, dst)

    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item, noc_route_table[route_item])

    route_table = {}
    for src in range (1000, 1000 + NOC_NODE_NUM * NOP_SIZE):
        for dst in range (1000, 1000 + NOC_NODE_NUM * NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst - 1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst, dst))
                route_table[(src,dst)].insert(0, (src, noc_src))

    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    noc_dict["Ring"] = Ring

    return noc_dict, ALL_SIM_NODE_NUM    

def construct_noc_nop_Ring(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    Ring = []
    NoC_h = int(NOC_NODE_NUM / NoC_w)

    for nop_id in range (NOP_SIZE): 
        for w in range(1, NoC_w):
            sub_Ring = []
            for y in range(NoC_h):
                id = NOC_NODE_NUM * nop_id + y * NoC_w + 0
                sub_Ring.append(id)
            
            for x in range(1, w):
                id = NOC_NODE_NUM * nop_id + (NoC_h - 1) * NoC_w + x
                sub_Ring.append(id)

            for y in range(NoC_h - 1, -1, -1):
                id = NOC_NODE_NUM * nop_id + y * NoC_w + w
                sub_Ring.append(id)
            
            for x in range(w - 1, 0, -1):
                id = NOC_NODE_NUM * nop_id + 0 * NoC_w + x
                sub_Ring.append(id)

            Ring.append(sub_Ring)        

    print('Ring = ', Ring)

    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local, src)] = 0
            F[(src, local)] = 0
            bw_scales[(local, src)] = 1
            bw_scales[(src, local)] = 1
            energy_ratio[(local, src)] = NOC_energy_ratio
            energy_ratio[(src, local)] = NOC_energy_ratio

    for sub_Ring in Ring:
        for i in range(len(sub_Ring)):
            F[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] = 0
            energy_ratio[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] = NOC_energy_ratio

            if (sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)]) in bw_scales.keys():
                bw_scales[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] += 1
            else:
                bw_scales[(sub_Ring[i], sub_Ring[(i + 1) % len(sub_Ring)])] = 1

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous Routerless ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)

    for src in range (0, NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0, NOC_NODE_NUM*NOP_SIZE):
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src

            if chip_id(cur_src) != chip_id(dst):
                if (src != 0) or ((noc_id(dst) % NoC_w) != 0):
                    continue
                
                sub_Ring = find_shortest_ring(Ring, src, comm_id(src))
                noc_route_table[(src, dst)] = append_ring_route(noc_route_table[(src, dst)], sub_Ring, src, comm_id(src))
                    
                # go to the nop node
                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src, dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

            if((noc_id(src) % NoC_w) == 0) or ((noc_id(dst) % NoC_w) == 0):
                if cur_src != dst:
                    sub_Ring = find_shortest_ring(Ring, cur_src, dst)
                    noc_route_table[(src, dst)] = append_ring_route(noc_route_table[(src, dst)], sub_Ring, cur_src, dst)

    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item, noc_route_table[route_item])

    route_table = {}
    for src in range (1000, 1000 + NOC_NODE_NUM * NOP_SIZE):
        for dst in range (1000, 1000 + NOC_NODE_NUM * NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst - 1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst, dst))
                route_table[(src,dst)].insert(0, (src, noc_src))

    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    noc_dict["Ring"] = Ring

    return noc_dict, ALL_SIM_NODE_NUM    

def construct_noc_nop_Routerless(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    NoC_h = NOC_NODE_NUM / NoC_w

    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = 1
            bw_scales[(src,local)] = 1
            energy_ratio[(local,src)] = NOC_energy_ratio
            energy_ratio[(src,local)] = NOC_energy_ratio
            src_x = src_local_id %  NoC_w
            src_y = int(src_local_id / NoC_w)

            for dst_local_id in range (NOC_NODE_NUM):
                dst = dst_local_id + NOC_NODE_NUM * nop_id
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_y == dst_y):
                    if (src_y == 0):
                        if (src_x - dst_x == 1):
                            F[(src, dst)] = 0
                            bw_scales[(src, dst)] = NoC_h - 1 + NoC_w - src_x
                            energy_ratio[(src,dst)] = NOC_energy_ratio
                    elif (src_y == NoC_h - 1):
                        if (dst_x - src_x == 1):
                            F[(src, dst)] = 0
                            bw_scales[(src, dst)] = 1 + NoC_w - dst_x
                            energy_ratio[(src,dst)] = NOC_energy_ratio
                    else:
                        if (dst_x - src_x == 1):
                            F[(src, dst)] = 0
                            bw_scales[(src, dst)] = 1
                            energy_ratio[(src,dst)] = NOC_energy_ratio

                elif (src_x == dst_x):
                    if (src_x == 0):
                        if (dst_y - src_y == 1):
                            F[(src, dst)] = 0
                            bw_scales[(src,dst)] = NoC_w - 1 + NoC_h - dst_y
                            energy_ratio[(src,dst)] = NOC_energy_ratio

                    elif (src_x == NoC_w - 1):
                        if (src_y - dst_y == 1):
                            F[(src, dst)] = 0
                            bw_scales[(src,dst)] = 1 + NoC_h - src_y
                            energy_ratio[(src,dst)] = NOC_energy_ratio
                    else:
                        if (src_y - dst_y == 1):
                            F[(src, dst)] = 0
                            bw_scales[(src,dst)] = 1
                            energy_ratio[(src,dst)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous Routerless ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)

    for src in range (0, NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0, NOC_NODE_NUM*NOP_SIZE):
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src

            if chip_id(cur_src) != chip_id(dst):
                if (src != 0) or ((noc_id(dst) % NoC_w) != 0):
                    continue
                
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)

                    if (src_noc_y > dst_noc_y): # go south
                        cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                    elif (src_noc_x > dst_noc_x):  # go west
                        cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                    else:
                        raise NotImplementedError
                    
                    cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                    
                # go to the nop node
                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst
                
            while cur_src != dst:
                src_noc_x = noc_id(cur_src)  %  NoC_w
                src_noc_y = int(noc_id(cur_src)  / NoC_w)
                dst_noc_x = noc_id(dst) %  NoC_w
                dst_noc_y = int(noc_id(dst) / NoC_w)

                if (noc_id(src) // NoC_w == 1) and (noc_id(src) % NoC_w == 0): # input
                    if (dst_noc_y == 0):
                        while(noc_id(cur_src) % NoC_w != NoC_w - 1): # east
                            noc_route_table[(src, dst)].append((cur_src, cur_src + 1))
                            cur_src = cur_src + 1
                        while(noc_id(cur_src) // NoC_w != dst_noc_y): # sorth
                            noc_route_table[(src, dst)].append((cur_src, cur_src - NoC_w))
                            cur_src = cur_src - NoC_w
                        while(noc_id(cur_src) % NoC_w != dst_noc_x): # west
                            noc_route_table[(src, dst)].append((cur_src, cur_src - 1))
                            cur_src = cur_src - 1
                    else:
                        while(noc_id(cur_src) // NoC_w != dst_noc_y): # north
                            noc_route_table[(src, dst)].append((cur_src, cur_src + NoC_w))
                            cur_src = cur_src + NoC_w
                        while(noc_id(cur_src) % NoC_w != dst_noc_x): # east
                            noc_route_table[(src, dst)].append((cur_src, cur_src + 1))
                            cur_src = cur_src + 1

                elif (noc_id(src) // NoC_w == 2) and (noc_id(src) % NoC_w == 0): # wgt
                    while(noc_id(cur_src) // NoC_w != NoC_h - 1): # north
                        noc_route_table[(src, dst)].append((cur_src, cur_src + NoC_w))
                        cur_src = cur_src + NoC_w

                    while(noc_id(cur_src) % NoC_w != dst_noc_x): # east
                        noc_route_table[(src, dst)].append((cur_src, cur_src + 1))
                        cur_src = cur_src + 1

                    while(noc_id(cur_src) // NoC_w != dst_noc_y): # sorth
                        noc_route_table[(src, dst)].append((cur_src, cur_src - NoC_w))
                        cur_src = cur_src - NoC_w

                elif (noc_id(dst) // NoC_w == 0) and (noc_id(dst) % NoC_w == 0): # out
                    while(noc_id(cur_src) // NoC_w != dst_noc_y): # sorth
                        noc_route_table[(src, dst)].append((cur_src, cur_src - NoC_w))
                        cur_src = cur_src - NoC_w

                    while(noc_id(cur_src) % NoC_w != dst_noc_x): # west
                        noc_route_table[(src, dst)].append((cur_src, cur_src - 1))
                        cur_src = cur_src - 1

                elif (src == 0) and (nop_id(dst) % NoC_w == 0):
                    while(noc_id(cur_src) // NoC_w != dst_noc_y): # north
                        noc_route_table[(src, dst)].append((cur_src, cur_src + NoC_w))
                        cur_src = cur_src + NoC_w
                else:
                    break

    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

    route_table = {}
    for src in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
        for dst in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst -1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst,dst))
                route_table[(src,dst)].insert(0,(src,noc_src))

    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    return noc_dict, ALL_SIM_NODE_NUM    


def construct_noc_nop_Torus(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    NoC_h = NOC_NODE_NUM / NoC_w

    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = 1
            bw_scales[(src,local)] = 1
            energy_ratio[(local,src)] = NOC_energy_ratio
            energy_ratio[(src,local)] = NOC_energy_ratio
            src_x = src_local_id %  NoC_w
            src_y = int(src_local_id / NoC_w)

            for dst_local_id in range (NOC_NODE_NUM):
                dst = dst_local_id + NOC_NODE_NUM * nop_id
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_x == dst_x) :
                    if (abs(src_y - dst_y) == 1) or (abs(src_y - dst_y) == NoC_h - 1) :
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio
                elif (src_y == dst_y) :
                    if (abs(src_x - dst_x) == 1) or (abs(src_x - dst_x) == NoC_w - 1):
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (abs(src_y - dst_y) == 1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (abs(src_x - dst_x) == 1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous Torus ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)
  
    for src in range (0,NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0,NOC_NODE_NUM*NOP_SIZE):
            # print ("src = ",src,"dst = ",dst)
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src
            if chip_id(cur_src) != chip_id(dst):
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                    # print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
                    if (src_noc_x > dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2):  # go west
                        cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_x > dst_noc_x):
                        cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2): # go east
                        cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x):
                        cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_y < dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go north
                        cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w
                    elif (src_noc_y < dst_noc_y):
                        cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                    elif (src_noc_y > dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go south
                        cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                    elif (src_noc_y > dst_noc_y):
                        cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w

                    cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                    
                # go to the nop node
                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

            while cur_src != dst:
                src_noc_x = noc_id(cur_src)  %  NoC_w
                src_noc_y = int(noc_id(cur_src)  / NoC_w)
                dst_noc_x = noc_id(dst) %  NoC_w
                dst_noc_y = int(noc_id(dst)  / NoC_w)
                # print ("src_x",src_x,"src_y",src_y,"dst_x",dst_x,"dst_y",dst_y)
                if (src_noc_x > dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2): # go west
                    cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w
                elif (src_noc_x > dst_noc_x):
                    cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                elif (src_noc_x < dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2): # go east
                    cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                elif (src_noc_x < dst_noc_x):
                    cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w

                elif (src_noc_y < dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go north
                    cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w
                elif (src_noc_y < dst_noc_y):
                    cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                elif (src_noc_y > dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go south
                    cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                else:
                    cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w

                cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

    route_table = {}
    for src in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
        for dst in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst -1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst,dst))
                route_table[(src,dst)].insert(0,(src,noc_src))


    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"]= energy_ratio 
    return noc_dict,ALL_SIM_NODE_NUM    

def construct_noc_nop_CMesh(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    NoC_h = NOC_NODE_NUM / NoC_w
    print('CORE_NUM = ', CORE_NUM)
    print('NOP_SIZE = ', NOP_SIZE)

    cmesh_dict = {2:[1, 6, 7], 4:[3, 8, 9], 12:[11, 16, 17], 14:[13, 18, 19]}
    cmesh_list = [0, 5, 10, 2, 4, 12, 14] 

    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local, src)] = 0
            F[(src, local)] = 0
            bw_scales[(local, src)] = 1
            bw_scales[(src, local)] = 1
            energy_ratio[(local, src)] = NOC_energy_ratio
            energy_ratio[(src, local)] = NOC_energy_ratio
            src_x = src_local_id %  NoC_w
            src_y = int(src_local_id / NoC_w)

            if src_local_id in cmesh_dict.keys():
                for dst_local_id in cmesh_dict[src_local_id]:
                    dst = dst_local_id + NOC_NODE_NUM * nop_id
                    F[(src, dst)] = 0
                    bw_scales[(src, dst)] = 1
                    energy_ratio[(src, dst)] = NOC_energy_ratio

                    F[(dst, src)] = 0
                    bw_scales[(dst, src)] = 1
                    energy_ratio[(dst, src)] = NOC_energy_ratio

                for dst_local_id in cmesh_list:
                    dst = dst_local_id + NOC_NODE_NUM * nop_id
                    dst_x = dst_local_id %  NoC_w
                    dst_y = dst_local_id // NoC_w
                    if (src_x == dst_x) :
                        if abs(dst_y - src_y) <= 2:
                            F[(src, dst)] = 0
                            bw_scales[(src, dst)] = 1
                            energy_ratio[(src, dst)] = NOC_energy_ratio

                            F[(dst, src)] = 0
                            bw_scales[(dst, src)] = 1
                            energy_ratio[(dst, src)] = NOC_energy_ratio
                    elif (src_y == dst_y) :
                        if abs(src_x - dst_x) <= 2:
                            F[(src, dst)] = 0
                            bw_scales[(src, dst)] = 1
                            energy_ratio[(src,dst)] = NOC_energy_ratio

                            F[(dst, src)] = 0
                            bw_scales[(dst, src)] = 1
                            energy_ratio[(dst, src)] = NOC_energy_ratio
                
        F[(5 + NOC_NODE_NUM * nop_id, 2 + NOC_NODE_NUM * nop_id)] = 0
        F[(5 + NOC_NODE_NUM * nop_id, 12 + NOC_NODE_NUM * nop_id)] = 0
        bw_scales[(5 + NOC_NODE_NUM * nop_id, 2 + NOC_NODE_NUM * nop_id)] = 1
        bw_scales[(5 + NOC_NODE_NUM * nop_id, 12 + NOC_NODE_NUM * nop_id)] = 1
        energy_ratio[(5 + NOC_NODE_NUM * nop_id, 2 + NOC_NODE_NUM * nop_id)] = NOC_energy_ratio
        energy_ratio[(5 + NOC_NODE_NUM * nop_id, 12 + NOC_NODE_NUM * nop_id)] = NOC_energy_ratio

        F[(5 + NOC_NODE_NUM * nop_id, 10 + NOC_NODE_NUM * nop_id)] = 0
        F[(10 + NOC_NODE_NUM * nop_id, 5 + NOC_NODE_NUM * nop_id)] = 0
        bw_scales[(5 + NOC_NODE_NUM * nop_id, 10 + NOC_NODE_NUM * nop_id)] = 1
        bw_scales[(10 + NOC_NODE_NUM * nop_id, 5 + NOC_NODE_NUM * nop_id)] = 1
        energy_ratio[(5 + NOC_NODE_NUM * nop_id, 10 + NOC_NODE_NUM * nop_id)] = NOC_energy_ratio
        energy_ratio[(10 + NOC_NODE_NUM * nop_id, 5 + NOC_NODE_NUM * nop_id)] = NOC_energy_ratio

        F[(5 + NOC_NODE_NUM * nop_id, 0 + NOC_NODE_NUM * nop_id)] = 0
        F[(0 + NOC_NODE_NUM * nop_id, 5 + NOC_NODE_NUM * nop_id)] = 0
        bw_scales[(5 + NOC_NODE_NUM * nop_id, 0 + NOC_NODE_NUM * nop_id)] = 1
        bw_scales[(0 + NOC_NODE_NUM * nop_id, 5 + NOC_NODE_NUM * nop_id)] = 1
        energy_ratio[(5 + NOC_NODE_NUM * nop_id, 0 + NOC_NODE_NUM * nop_id)] = NOC_energy_ratio
        energy_ratio[(0 + NOC_NODE_NUM * nop_id, 5 + NOC_NODE_NUM * nop_id)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous mesh ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)

    def append_cmesh_route(route, src, dst):
        # print(src, dst)
        # print('noc_id(src) = ', noc_id(src))
        # print('noc_id(dst) = ', noc_id(dst))
                    
        if noc_id(src) in cmesh_list:
            src_router = src
        else:
            for key in cmesh_dict.keys():
                if noc_id(src) in cmesh_dict[key]:
                    src_router = key + chip_id(src) * NOC_NODE_NUM
                    # print('cmesh_dict[key] = ', cmesh_dict[key])
                    break
        
        if noc_id(dst) in cmesh_list:
            dst_router = dst
        else:
            for key in cmesh_dict.keys():
                if noc_id(dst) in cmesh_dict[key]:
                    dst_router = key + chip_id(dst) * NOC_NODE_NUM
                    break

        if src_router == dst_router:
            route.append((src, src_router))
            route.append((src_router, dst))
            return route
        
        if src != src_router:
            route.append((src, src_router))
        cur_src = src_router

        while cur_src != dst_router: # go to the first noc node
            src_noc_x = noc_id(cur_src) %  NoC_w
            src_noc_y = noc_id(cur_src) // NoC_w

            dst_noc_x = noc_id(dst_router) % NoC_w
            dst_noc_y = noc_id(dst_router) // NoC_w

            if (src_noc_x > dst_noc_x):  # go west
                cur_noc_dst = src_noc_x - 1 + src_noc_y * NoC_w
                if noc_id(cur_noc_dst) not in cmesh_list:
                    cur_noc_dst = src_noc_x - 2 + src_noc_y * NoC_w

            elif (src_noc_x < dst_noc_x): # go east
                cur_noc_dst = src_noc_x + 1 +  src_noc_y * NoC_w
                if (noc_id(cur_noc_dst) not in cmesh_list) and noc_id(cur_src) != 5:
                    cur_noc_dst = src_noc_x + 2 + src_noc_y * NoC_w
                elif src_noc_y < dst_noc_y:
                    cur_noc_dst = 10
                else:
                    cur_noc_dst = 0

            elif (src_noc_y < dst_noc_y): # go north
                cur_noc_dst = src_noc_x + (src_noc_y + 1) * NoC_w
                if noc_id(cur_noc_dst) not in cmesh_list:
                    cur_noc_dst = src_noc_x + (src_noc_y + 2) * NoC_w

            elif (src_noc_y > dst_noc_y): # go south
                cur_noc_dst = src_noc_x + (src_noc_y - 1) * NoC_w
                if noc_id(cur_noc_dst) not in cmesh_list:
                    cur_noc_dst = src_noc_x + (src_noc_y - 2) * NoC_w

            cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
            route.append((cur_src, cur_dst))
            cur_src = cur_dst

        if dst != dst_router:     
            route.append((dst_router, dst))
        return route
       
    for src in range (0, NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0, NOC_NODE_NUM*NOP_SIZE):
            # print("src = ",src,"dst = ",dst)
            noc_route_table[(src, dst)] = []
            cur_src = src
            cur_dst = src
            if chip_id(cur_src) != chip_id(dst):
                if (src != 0) or ((noc_id(dst) % NoC_w) != 0):
                    continue

                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src, dst)].append((cur_src, cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

            if noc_id(src) % NOC_NODE_NUM == 15 or noc_id(dst) % NOC_NODE_NUM == 15:
                continue
            if((noc_id(src) % NoC_w) == 0) or ((noc_id(dst) % NoC_w) == 0):
                if cur_src != dst:
                    noc_route_table[(src, dst)] = append_cmesh_route(noc_route_table[(src, dst)], cur_src, dst)
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

    route_table = {}
    for src in range (1000, 1000+NOC_NODE_NUM*NOP_SIZE):
        for dst in range (1000, 1000+NOC_NODE_NUM*NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst - 1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst, dst))
                route_table[(src,dst)].insert(0,(src, noc_src))

    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"] = route_table
    noc_dict["F"] = F
    noc_dict["bw_scales"] = bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    return noc_dict, ALL_SIM_NODE_NUM

def construct_noc_nop_Mesh(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    print('CORE_NUM = ', CORE_NUM)
    print('NOP_SIZE = ', NOP_SIZE)

    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):  
            for dst_local_id in range (NOC_NODE_NUM):
                src = src_local_id + NOC_NODE_NUM * nop_id
                dst = dst_local_id + NOC_NODE_NUM * nop_id
                local = src + 1000
                F[(local,src)] = 0
                F[(src,local)] = 0
                bw_scales[(local,src)] = 1
                bw_scales[(src,local)] = 1
                energy_ratio[(local,src)] = NOC_energy_ratio
                energy_ratio[(src,local)] = NOC_energy_ratio
                src_x = src_local_id %  NoC_w
                src_y = int(src_local_id / NoC_w)
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_x == dst_x) :
                    if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio
                elif (src_y == dst_y) :
                    if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))


    
    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous mesh ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)
       
    for src in range (0,NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0,NOC_NODE_NUM*NOP_SIZE):
            # print ("src = ",src,"dst = ",dst)
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src
            if chip_id(cur_src) != chip_id(dst):
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                    # print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
                    if (src_noc_x > dst_noc_x):  # go west
                        cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x): # go east
                        cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                    elif (src_noc_y < dst_noc_y): # go north
                        cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                    elif (src_noc_y > dst_noc_y): # go south
                        cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                    cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                    
                # go to the nop node
                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

            while cur_src != dst:
                src_noc_x = noc_id(cur_src)  %  NoC_w
                src_noc_y = int(noc_id(cur_src)  / NoC_w)
                dst_noc_x = noc_id(dst) %  NoC_w
                dst_noc_y = int(noc_id(dst)  / NoC_w)
                # print ("src_x",src_x,"src_y",src_y,"dst_x",dst_x,"dst_y",dst_y)
                if (src_noc_x > dst_noc_x): # go west
                    cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                elif (src_noc_x < dst_noc_x): # go east
                    cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                elif (src_noc_y < dst_noc_y): # go north
                    cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                elif (src_noc_y > dst_noc_y): # go south
                    cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

    route_table = {}
    for src in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
        for dst in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst -1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst,dst))
                route_table[(src,dst)].insert(0,(src,noc_src))


    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"] = route_table
    noc_dict["F"] = F
    noc_dict["bw_scales"] = bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    return noc_dict, ALL_SIM_NODE_NUM 

def construct_noc_nop_Bus(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    print('CORE_NUM = ', CORE_NUM)
    print('NOP_SIZE = ', NOP_SIZE)
    w_l2_node_offset = NoC_w * 2 
    bus_node_src_offset = 2000
    bus_node_dst_offset = 2001

    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        bus_node_src = bus_node_src_offset + NOC_NODE_NUM * nop_id
        bus_node_dst = bus_node_dst_offset + NOC_NODE_NUM * nop_id
        w_l2_node = w_l2_node_offset + NOC_NODE_NUM * nop_id

        F[(bus_node_src,bus_node_dst)] = 0
        bw_scales[(bus_node_src,bus_node_dst)] = 4
        energy_ratio[(bus_node_src,bus_node_dst)] = NOC_energy_ratio

        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000

            F[(local,src)] = 0
            F[(src,local)] = 0
            if src == w_l2_node:
                bw_scales[(local,src)] = nop_scale_ratio
                bw_scales[(src,local)] = nop_scale_ratio
                energy_ratio[(local,src)] = DIE2DIE_energy_ratio + DRAM_energy_ratio
                energy_ratio[(src,local)] = DIE2DIE_energy_ratio + DRAM_energy_ratio
            else:
                bw_scales[(local,src)] = 1
                bw_scales[(src,local)] = 1
                energy_ratio[(local,src)] = NOC_energy_ratio
                energy_ratio[(src,local)] = NOC_energy_ratio

            F[(src,bus_node_src)] = 0
            F[(bus_node_dst,src)] = 0
            bw_scales[(src,bus_node_src)] = 1
            bw_scales[(bus_node_dst,src)] = 1
            energy_ratio[(src,bus_node_src)] = NOC_energy_ratio
            energy_ratio[(bus_node_dst,src)] = NOC_energy_ratio

        
    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))


    
    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous mesh ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)
       
    for src in range (0,NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0,NOC_NODE_NUM*NOP_SIZE):
            # print ("src = ",src,"dst = ",dst)
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src
            if chip_id(cur_src) != chip_id(dst):
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                    # print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
                    if (src_noc_x > dst_noc_x):  # go west
                        cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x): # go east
                        cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                    elif (src_noc_y < dst_noc_y): # go north
                        cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                    elif (src_noc_y > dst_noc_y): # go south
                        cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                    cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                    
                # go to the nop node
                cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

                while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                    src_nop_x = nop_id(cur_src) % NoP_w
                    src_nop_y = int (nop_id(cur_src) / NoP_w)
                    dst_nop_x = chip_id(dst) % NoP_w
                    dst_nop_y = int(chip_id(dst) / NoP_w)
                    if (src_nop_x > dst_nop_x):  # go west
                        cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                    elif (src_nop_x < dst_nop_x): # go east
                        cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                    elif (src_nop_y < dst_nop_y): # go north
                        cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                    elif (src_nop_y > dst_nop_y): # go south
                        cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                    cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                    noc_route_table[(src,dst)].append((cur_src,cur_dst))
                    cur_src = cur_dst
                
                # go to the communication id 
                cur_dst = chip_id(dst) * NOC_NODE_NUM
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst

            chiplet_id = chip_id(dst)
            bus_node_src = bus_node_src_offset + chiplet_id * NOC_NODE_NUM
            bus_node_dst = bus_node_dst_offset + chiplet_id * NOC_NODE_NUM
            noc_route_table[(src,dst)].append((cur_src,bus_node_src))
            noc_route_table[(src,dst)].append((bus_node_src,bus_node_dst))
            noc_route_table[(src,dst)].append((bus_node_dst,dst))
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

    route_table = {}
    for src in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
        for dst in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst -1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst,dst))
                route_table[(src,dst)].insert(0,(src,noc_src))


    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"] = route_table
    noc_dict["F"] = F
    noc_dict["bw_scales"] = bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    #print("route_table ",route_table)
    print("bw_scales ",bw_scales)
    #print("energy_ratio ",energy_ratio)
    return noc_dict, ALL_SIM_NODE_NUM   

def construct_noc_nop_Bus_1(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    NOP_SIZE = 1
    CORE_NUM = NOC_NODE_NUM  *NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE

    L2_w_node = NoC_w * 2


    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        bus_1 = 0 + 2000 + NOC_NODE_NUM * nop_id
        bus_2 = 1 + 2000 + NOC_NODE_NUM * nop_id
        F[(bus_1,bus_2)] = 0
        bw_scales[(bus_1,bus_2)] = 4 / 32
        energy_ratio[(bus_1,bus_2)] = NOC_energy_ratio
        for src_local_id in range (NOC_NODE_NUM):  
            src = src_local_id + NOC_NODE_NUM * nop_id  + 10
            bus_1 = 0 + 2000 + NOC_NODE_NUM * nop_id
            bus_2 = 1 + 2000 + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = 1
            bw_scales[(src,local)] = 1
            if src_local_id == L2_w_node:
                energy_ratio[(local,src)] = DIE2DIE_energy_ratio + DRAM_energy_ratio
                energy_ratio[(src,local)] = DIE2DIE_energy_ratio + DRAM_energy_ratio
            else:
                energy_ratio[(local,src)] = 0
                energy_ratio[(src,local)] = 0
            F[(bus_2,src)] = 0
            F[(src,bus_1)] = 0
            bw_scales[(bus_2,src)] = 1
            bw_scales[(src,bus_1)] = 1
            energy_ratio[(bus_2,src)] = NOC_energy_ratio
            energy_ratio[(src,bus_1)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id +10
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id +10
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE + 100
        bus_1 = 0 + 2000 + NOC_NODE_NUM * nop_id
        bus_2 = 1 + 2000 + NOC_NODE_NUM * nop_id
        F[(bus_2,nop_router_id)] = 0
        F[(nop_router_id,bus_1)] = 0
        bw_scales[(bus_2,nop_router_id)] = 1
        bw_scales[(nop_router_id,bus_1)] = 1
        energy_ratio[(bus_2,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,bus_1)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))


    
    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous bus ---- \n\n")

    noc_route_table = {}
    hops = {}

    def noc_id (real_id):
        return (real_id % NOC_NODE_NUM)

    def chip_id (real_id):
        return (int (real_id /NOC_NODE_NUM) )

    def comm_id (real_id): # the communication router id in one chip
        return (chip_id(real_id)*NOC_NODE_NUM)

    def nop_id (real_id):
        return (real_id - NOC_NODE_NUM*NOP_SIZE)
       
    for src in range (0,NOC_NODE_NUM*NOP_SIZE):
        src = src + 10
        for dst in range (0,NOC_NODE_NUM*NOP_SIZE):
            # print ("src = ",src,"dst = ",dst)
            dst = dst + 10
            noc_route_table[(src,dst)] = []

            bus_1 = 0 + 2000 + NOC_NODE_NUM * 0
            bus_2 = 1 + 2000 + NOC_NODE_NUM * 0
            noc_route_table[(src,dst)].append((src,bus_1))
            noc_route_table[(src,dst)].append((bus_1,bus_2))
            noc_route_table[(src,dst)].append((bus_2,dst))
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

    route_table = {}
    for src in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
        src += 10
        for dst in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
            dst += 10
            route_table[(src,dst)] = []
            noc_src = src - 1000
            noc_dst = dst -1000
            route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
            if (src!=dst):
                route_table[(src,dst)].append((noc_dst,dst))
                route_table[(src,dst)].insert(0,(src,noc_src))


    for item in route_table:
        hops[item] = len(route_table[item])
        # print (item,route_table[item])

    print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
    noc_dict["route_table"] = route_table
    noc_dict["F"] = F
    noc_dict["bw_scales"] = bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    print("route_table ",route_table)
    print("bw_scales ",bw_scales)
    print("energy_ratio ",energy_ratio)
    return noc_dict, ALL_SIM_NODE_NUM    


def construct_noc_nop_topo(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio, topology = 'Mesh', constraint = 5, ol2_node = 0):
    if topology == 'Mesh':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Mesh(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'CMesh':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_CMesh(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'Torus':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Torus(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'Routerless':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Routerless(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'Ring':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Ring(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'RandomRouterless':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_RandomRouterless(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio, NoC_w, ol2_node)
    elif topology == 'Bus':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Bus(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    else:
        raise NotImplementedError

    #assert len(noc_dict["route_table"]) == (NOC_NODE_NUM * NOP_SIZE) ** 2, "len(noc_dict['route_table']) = %d which is wrong" % len(noc_dict['route_table'])
    #assert len(noc_dict['F']) == len(noc_dict['bw_scales']), "len(noc_dict['F']) != len(noc_dict['bw_scales'])"

    return noc_dict, ALL_SIM_NODE_NUM

if __name__ == '__main__':
    #TOPO_param = {"NoC_w":5, "NOC_NODE_NUM": 20, "NoP_w": 3, "NOP_SIZE": 6, "nop_scale_ratio":0.5, "topology": 'CMesh'}
    #NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = TOPO_param["topology"])
    NoC_param, all_sim_node_num = construct_noc_nop_Bus(40, 5, 2, 2, 0.5)