# 用于实现各种多播方案，会被性能预测代码使用，确定每组communication中需要途径的link

import math
import os
import sys
import random
import numpy as np

def simple_multicast(src,dst_list, route_table):
    link_set = set()
    for dst in dst_list:
        for link in route_table[src,dst]:
            link_set.add(link)
    return link_set