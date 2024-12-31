
from os.path import join, isfile, basename
from os import replace
import re
from utils import get_root_path, create_dir_if_not_exists
import redis
from typing import Dict, List, Union
from glob import iglob
import pickle5 as pickle

DesignPoint = Dict[str, Union[int, str]]

def update_csv_dict(csv_dict, kernel, tool, framework, board, synthesis, BENCHMARK, obj):
    if csv_dict is not None:
        csv_dict[kernel] = {'kernel': kernel}
        csv_dict[kernel]['xilinx tool'] = tool
        csv_dict[kernel]['framework'] = framework
        csv_dict[kernel]['benchmark'] = BENCHMARK
        csv_dict[kernel]['board'] = board
        if synthesis: csv_dict[kernel]['synthesis or on-board'] = 'synthesis'
        else: csv_dict[kernel]['synthesis or on-board'] = 'on-board'
        csv_dict[kernel]['point'] = str(obj.point.values())
        csv_dict[kernel]['latency'] = obj.perf
        for k, u in obj.res_util.items():
            csv_dict[kernel][k] = u
        
                
                
def log_dict_of_dicts_to_csv(fn, csv_dict, csv_header, delimiter=','):
    import csv
    fp = open(fn, 'w+')
    f_writer = csv.DictWriter(fp, fieldnames=csv_header)
    f_writer.writeheader()
    for d, value in sorted(csv_dict.items()):
        if d == 'header':
            continue
        f_writer.writerow(value)
    fp.close()   

def apply_design_point(src_path: List, db_path, dest_path, VER, tool, framework, board, synthesis, BENCHMARK) -> bool:
    src_files = [f for f in iglob(join(src_path, '**/*'), recursive=True) if f.endswith('.c') and 'large-size' not in f]
    for file_name in src_files:
        print(file_name)
        if isfile(file_name):
            KERNEL = basename(file_name).split('_')[0]
            obj = get_best_design(db_path, KERNEL, VER)
            if obj is None:
                continue
            point = obj.point
            update_csv_dict(csv_dict, KERNEL, tool, framework, board, synthesis, BENCHMARK, obj)
            with open(file_name, 'r', errors='replace') as src_file, \
                    open('{0}/applier_temp.txt'.format(dest_path), 'w', errors='replace') as dest_file:
                for line in src_file:
                    # Assume one pragma per line even though a loop here.
                    if line.startswith('#include "merlin_type_define.h"'):
                        continue
                    for auto, ds_id in re.findall(r'(auto{(.*?)})', line, re.IGNORECASE):
                        if ds_id not in point:
                            print('Parameter %s not found in design point', ds_id)
                        else:
                            # Replace "auto{?}" with a specific value                              
                            line = line.replace(auto, str(point[ds_id]))
                    dest_file.write(line)
            replace('{0}/applier_temp.txt'.format(dest_path),
                        '{0}/{1}'.format(dest_path, basename(file_name)))


# run "redis-server" on command line first!
def get_best_design(db_path, KERNEL, VER, mode='min'):
    # create a redis database
    database = redis.StrictRedis(host='localhost', port=6379)
    database.flushdb()
    round_num = 3 if VER == 'v20' else 13
    db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and f'{KERNEL}_' in f and 'large-size' not in f and VER in f and f'one-db-extended-round{round_num}' in f and 'archive' not in f and 'single-merged' not in f] 
    print(db_files)
    if len(db_files) == 0:
        print(f'Warning: no database found for kernel {KERNEL} in ver {VER}')
        return
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    for idx, file in enumerate(db_files):
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        database.hmset(0, data)
        max_idx = idx + 1
    keys = [k.decode('utf-8') for k in database.hkeys(0)]

    keys = sorted(keys)
    points = {}
    ret_point = None
    min_perf = float('inf')
    for i in range(len(keys)):
        pickle_obj = database.hget(0, keys[i])
        obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
        if type(obj) is int or type(obj) is dict:
            continue
        if keys[i][0:3] == 'lv1' or obj.ret_code.name == 'TIMEOUT' or obj.perf < 100.0:
            continue                    

        s = str(obj.point.values())
        utils = {k[5:]: max(0.0, u) for k, u in obj.res_util.items() if k.startswith('util-')}
        valid = all([utils[res] < 0.80001 for res in utils])
        if obj.perf != 0 and obj.perf < min_perf and valid:
            min_perf = obj.perf
            ret_point = obj
        # if s not in points:
        #     points[s] = (i, keys[i], obj.perf, obj.ret_code)


    if ret_point:
        return ret_point
    
    print(f'Error: no valid point found for {KERNEL}')
    return None


  
        
if __name__ == '__main__':
    for VER in ['v18', 'v20']:
        csv_dict = {'header' : ['kernel', 'benchmark', 'xilinx tool', 'board', 'synthesis or on-board', 'framework', 'latency', 'util-BRAM', 'util-DSP', 'util-FF', 'util-LUT', 'total-BRAM', 'total-DSP', 'total-FF', 'total-LUT', 'point']}
        tool = 'SDx18.3' if VER == 'v18' else 'Vitis20.2'
        framework = 'GNN-DSE'
        board = 'xilinx_u200_xdma_201830_2'
        synthesis = True
        for BENCHMARK in ['machsuite', 'poly']:
            src_path = join(get_root_path(), BENCHMARK, 'sources')
            dest_path = join(get_root_path(), VER, BENCHMARK, 'sources')
            create_dir_if_not_exists(dest_path)
            db_path = join(get_root_path(), BENCHMARK, 'databases/**/*')
            apply_design_point(src_path, db_path, dest_path, VER, tool, framework, board, synthesis, BENCHMARK)
        log_dict_of_dicts_to_csv(join(get_root_path(), VER, f'result_summary.csv'), csv_dict, csv_dict['header'])
