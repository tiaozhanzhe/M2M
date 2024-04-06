

def extract(file):
    file_i = open(file, 'r')
    
    lines = file_i.readlines()
    
    name_list = []
    tick_list = []
    DR_list   = []
    
    for line in lines:
        if line.startswith("workload"):
            item = line.split("DR:")
            DR = int(item[1].replace(";",'').replace("\n",''))
            name_item = item[0].split("/")
            name = name_item[-1]
            name_list.append(name)
            DR_list.append(DR)
        elif line.startswith("----- finish_node_num"):
            item = line.split("final_tick=")
            tick = int(item[1].replace(";",'').replace("\n",''))
            tick_list.append(tick)      
    file_i.close()
    
    out_line = "name\t tick\t DR\n"
    
    for i in range(len(DR_list)):
        name = name_list[i]
        tick = tick_list[i]
        DR   = DR_list[i]
        
        out_line += "{}\t{}\t{}\n".format(name, tick, DR)

    file = './gem5_result.txt'
    file_o = open(file, 'w')
    print(out_line, file=file_o)
    file_o.close()
    
if __name__ == '__main__':
    extract('/home/zhangjm/python/old/hetero_gem5/gem5/gem5_o_record_wxy.txt')
    # extract('../gem5/gem5_o_record_wxy.txt')