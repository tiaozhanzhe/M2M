

import os
import sys

if __name__ == '__main__':
    workload_name = sys.argv[1]
    abs_dir = os.path.dirname(os.path.realpath(__file__))
    
    print("result extract start------------")
    # record tick
    log_file = os.path.join(abs_dir, "wxy.log")
    log_f = open(log_file, 'r')
    record_line = ""
    lines = log_f.readlines()
    node_num = 0
    final_tick = 0
    for line in reversed(lines):
        if line.startswith("Exiting") or line == "\n" or line == "":
            pass
        elif line.startswith("node"):
            item = line.split(" ")
            node_id = item[1]
            tick = item[5]
            out_line = "node({}):{};\t".format(node_id, tick)
            record_line += out_line
            if node_num == 0:
                final_tick = tick
            node_num += 1
        else:
            break
    log_f.close()
    
    # record DR
    dr_file = os.path.join(abs_dir, "dnn_out/gem5_link_DR/gem5_link_DR_record.txt")
    dr_f = open(dr_file, 'r+')
    lines = dr_f.readlines()
    dr = 0
    print (lines)
    for line in reversed(lines):
        if line == "" or line == "\n":
            pass
        else:
            dr = int(line)
            print("\t{} recv--".format(dr), file = dr_f)
            break
    dr_f.close()
    
    out_file = os.path.join(abs_dir, "gem5_o_record_wxy.txt")
    out_f = open(out_file, 'a')
    print("workload: {};\t DR: {};".format(workload_name, dr), file = out_f)
    print("----- finish_node_num={}; final_tick={};".format(node_num, final_tick), file = out_f)
    print("----- {};".format(record_line), file = out_f)
    out_f.close()
    
            
            
    