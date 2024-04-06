

import os
import sys

if __name__ == '__main__':
    workload_name = sys.argv[1]
    abs_dir = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(abs_dir, "wxy.log")
    log_f = open(log_file, 'r')
    
    lines = log_f.readlines()
    for line in reversed(lines):
        if line.startswith("node"):
            item = line.split(" ")
            node_id = item[1]
            tick = item[5]
            break
    log_f.close()
    
    out_file = os.path.join(abs_dir, "grm5_o_record_wxy.txt")
    out_f = open(out_file, 'a')
    print("workload: {};\t tick: {};\t last_node: {};".format(workload_name, tick, node_id), file = out_f)
    out_f.close()
    
            
            
    