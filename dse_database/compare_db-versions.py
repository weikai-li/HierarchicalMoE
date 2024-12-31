import redis
import pickle5 as pickle
from collections import Counter
from os.path import join
from glob import glob, iglob
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import csv

import matplotlib
font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 17}

matplotlib.rc('font', **font)

from utils import get_root_path, create_dir_if_not_exists, save_pickle, \
    _report_rmse_etc, plot_points_with_subplot


# DATASET = 'cnn1'
DATASET = 'machsuite'
# DATASET = 'poly'
if DATASET == 'cnn1':
    DIR = './save/cnn1'
    KERNEL = 'cnn'
    gexf_file = './new_db/case1/CnnKernel1.cpp.gexf'
    db_path = './databases/cnn_case1/'
elif DATASET == 'machsuite':
    DIR = './save/machsuite/gemm-ncubed'
    KERNEL = 'gemm-ncubed'
    gexf_file = './machsuite/dot-files/gemm-ncubed_kernel.cpp.gexf'
    db_path = './machsuite/databases/**/*'
    # db_path = './machsuite/databases/original-size/bottleneck/**/*'
elif DATASET == 'poly':
    db_path = './poly/databases/**/*'

COMMON_DIR = join(get_root_path(), 'machsuite/databases/common-original-size/')
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
                    'nw', 'md', 'stencil-3d']

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium']

objectives = ['perf', 'util-LUT', 'util-FF', 'util-BRAM', 'util-DSP']
objectives = ['perf', 'util-BRAM']
csv_dict = {'header' : ['gname', 'pragma', 'v18-perf', 'v20-perf', 'v18-util-LUT', 'v20-util-LUT', 
                        'v18-util-FF', 'v20-util-FF', 'v18-util-DSP', 'v20-util-DSP', 
                        'v18-util-BRAM', 'v20-util-BRAM']}

def init_db(port=6379):
    database = redis.StrictRedis(host='localhost', port=port)
    database.flushdb()
    return database

def store_db(database, db_id, db_file_path) -> bool:
    #pylint:disable=missing-docstring

    dump_db = {
        key: database.hget(db_id, key)
        for key in database.hgetall(db_id)
    }
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

    return True

def commit_db(database, db_id, key, result):
    pickled_result = pickle.dumps(result)
    database.hset(db_id, key, pickled_result)
    
def get_db_files(db_path, key_word1 = None, key_word2 = None):
    if key_word2:
        db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 
        key_word2 in f and ('archive' not in f) and ('large' not in f) and ('single-merged' not in f) and ('-5' not in f)
        and 'one-db-extended-round2' in f]
    else:
        db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_'  in f and 
        # 'v20' in f and ('archive' not in f) and ('large' not in f) and 'merged' in f]
        'v20' not in f and ('archive' not in f) and ('large' not in f) and 'single-merged' not in f and 'maml' not in f
        and 'one-db-extended-round12' in f]
    # db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if 'archive' in f]
    return db_files

def read_db(database, db_files, combine = True, idx_db = 0):
    '''
    combine: merge all the databases as one or read them separately
    '''
    # idx_db = 0
    for idx, file in enumerate(db_files):
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        database.hmset(idx_db, data)
        if not combine:
            idx_db += 1
        f_db.close()
    if combine: idx_db = 1
    return idx_db
            
def get_keys(database, num_db):
    all_keys = {}
    for n in range(num_db):
        keys = [k.decode('utf-8') for k in database.hkeys(n)]
        all_keys[n] = sorted(keys)
    return all_keys

def get_keys_with_id(database, id):
    all_keys = {}
    keys = [k.decode('utf-8') for k in database.hkeys(id)]
    all_keys[id] = sorted(keys)
    return all_keys[id]

def compare_points_and_keys(all_points, all_keys, first, second): 
    res = (all_points[first]).items() <= (all_points[second]).items()
    res_key = set(all_keys[first]).issubset(set(all_keys[second]))
    
    return res, res_key 
        

def log_dict_of_dicts_to_csv(fn, csv_dict, csv_header, delimiter=','):
    import csv
    fp = open(join('', f'{fn}.csv'), 'w+')
    f_writer = csv.DictWriter(fp, fieldnames=csv_header)
    f_writer.writeheader()
    for d, value in csv_dict.items():
        if d == 'header':
            continue
        f_writer.writerow(value)
    fp.close()
    
def read_csv_dict(fn, csv_header):
    import csv
    res = []
    with open(join('', f'{fn}.csv')) as fp:
        csv_obj = csv.DictReader(fp)
        for row in csv_obj:
            res.append(row)
            
    return res

def add_to_csv_obj(csv_dict, gname, pragma, obj_0, obj_1): ## obj_0: v20, obj_1: v18
    if f'{gname}-{pragma}' not in csv_dict:
        csv_dict[f'{gname}-{pragma}'] = {'gname': gname, 'pragma': pragma}
    for o in objectives:
        if o == 'perf':
            csv_dict[f'{gname}-{pragma}'][f'v18-{o}'] = obj_1.perf
            csv_dict[f'{gname}-{pragma}'][f'v20-{o}'] = obj_0.perf
        else:
            csv_dict[f'{gname}-{pragma}'][f'v18-{o}'] = obj_1.res_util[o]
            csv_dict[f'{gname}-{pragma}'][f'v20-{o}'] = obj_0.res_util[o]

# run "redis-server" on command line first!
def compare_db_object(db_path, key_word1 = None, key_word2 = 'v20', normalize_perf=False, store_common=True, COMMON_DIR = None):
    # create a redis database
    database = init_db()
    database2 = init_db(port=8888)
    FLAG = 'common'
    
    ## read v20
    db_files = get_db_files(db_path, key_word1, key_word2)
    if len(db_files) == 0:
        print('no v20 db found')
        return
    pprint(db_files)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    num_db = read_db(database, db_files, combine = True, idx_db = 0)

    ## read v18
    db_files = get_db_files(db_path, key_word1, None)
    if len(db_files) == 0:
        print('no v18 db found')
    pprint(db_files)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    num_db = read_db(database2, db_files, combine = True, idx_db = 0)
    all_keys_0 = get_keys_with_id(database, 0)
    all_keys_1 = get_keys_with_id(database2, 0)
    valid_keys_0 = [k for k in all_keys_0 if 'lv2' in k]
    valid_keys_1 = [k for k in all_keys_1 if 'lv2' in k]
    print(f'keys1: {len(all_keys_0)}, keys2: {len(all_keys_1)}')
    print(f'valid keys1: {len(valid_keys_0)}, valid keys2: {len(valid_keys_1)}')
    print(f'number of same keys: {len((set(all_keys_0)).intersection(all_keys_1))}')
    print(f'number of same valid keys: {len((set(valid_keys_0)).intersection(valid_keys_1))}')
    common_keys = (set(valid_keys_0)).intersection(valid_keys_1)
    #print(num_db)
    # for k in common_keys:
    #    if k not 

    
    all_points = {}
    all_perf_seen = {}
    all_perfs = {}
    iii = 0
    id = 1
    all_common_data = {}
    all_common = []
    obj_keys = ['point', 'latency', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF', 
                    'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']
    all_common.append(obj_keys)
    count1, count2 = 0, 0
    for key in all_keys_0:
        # print(f'number of keys for database {id}: {len(keys)}')
        # count1 += 1
        pickle_obj = database.hget(0, key)
        # obj_0 = pickle.loads(pickle_obj)
        obj_0 = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
        if type(obj_0) is int or type(obj_0) is dict:
                continue
        if store_common == False:
            if 'lv1' in key: # or (not obj_0.ret_code.name == 'PASS' and not obj_0.ret_code.name == 'TIMEOUT'):
                continue
        if pickle_obj is None:
            # print(f'key {i}:{keys[i]} yeilds none object')
            continue
        
        if store_csv:
            if 'lv1' in key or obj_0.perf == 0.0: 
                continue
        
        if database2.hexists(0, key):
            # print('exists')
            # count2 += 1
            # if key not in common_keys:
            #     print(key)
            pickle_obj_1 = database2.hget(0, key)
            if pickle_obj_1 is None:
            # print(f'key {i}:{keys[i]} yeilds none object')
                continue
            obj_1 = pickle.loads(pickle_obj_1.replace(b'localdse', b'autodse'))
            if store_common or store_csv:
                # print(obj_0.ret_code, obj_1.ret_code)
                # try:
                if obj_1.ret_code.name == obj_0.ret_code.name:
                    if store_common:
                        commit_db(database, 3, key, obj_0)
                    if store_csv:
                        add_to_csv_obj(csv_dict, key_word1, key, obj_0, obj_1)
                    if 'lv2' in key and obj_0.ret_code.name == 'PASS': count2 += 1
                # except:
                #     print(f'skipping {key} with obj_0.ret_code: {obj_0.ret_code} and obj_1.ret_code: {obj_1.ret_code}')
            # print(f'''{key} exists in both
            #         v20: {obj_0.perf}, {obj_0.res_util}, {obj_0.ret_code}
            #         v18: {obj_1.perf}, {obj_1.res_util}, {obj_1.ret_code} ''')
            # if obj_0.ret_code.name == 'TIMEOUT' or obj_1.ret_code.name == 'TIMEOUT':
            #     continue
            else:
                pass
                # all_common_data[key] = {'perf': [obj_0.perf, obj_1.perf],
                #                         'res': [obj_0.res_util, obj_1.res_util],
                #                         'ret': [obj_0.ret_code.name, obj_1.ret_code.name]}
                # current_key = [key, (obj_0.perf, obj_1.perf)]
                # for o in obj_keys:
                #     if o in obj_0.res_util:
                #         current_key.append((obj_0.res_util[o], obj_1.res_util[o]))
                # all_common.append(current_key)
            # s = str(obj.point.values())
            # if s not in points:
            #     points[s] = (keys[i], obj.perf, obj.ret_code.name)
            # else:
            #     diff_label_bool = obj.perf != points[s][2]
            #     if diff_label_bool:
            #         diff_label += 1
            # perf_seen[obj.perf] += 1
            # perfs.append(obj.perf)
        else:
            # print('not exist')
            if store_common:
                # print(obj_0.ret_code, obj_1.ret_code)
                try:
                    commit_db(database, 2, key, obj_0)
                except:
                    print(f'skipping {key} with obj_0.ret_code: {obj_0.ret_code}')
                    
        # all_points[id] = points
        # all_perf_seen[id] = perf_seen
        # all_perfs[id] = perfs
        # print(perf_seen)
        # print(iii)
        # print(f'max cycle count for {id} is {max(perf_seen.keys())} and min cycle count is {min(perf_seen.keys())}')
    print('################')
    print(key_word1, 'common', count2)
    print('################')
    if not store_csv:
        if not store_common:
            with open('common_ncubed_version.pickle', 'wb') as f:
                pickle.dump(all_common_data, f)
            with open('common_ncubed_version.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(all_common)
        else:
            if FLAG == 'common': idx = 3
            else: idx = 2
            all_keys_0 = get_keys_with_id(database, idx)
            print(f'{key_word1} had {len(all_keys_0)} keys in common')
            if len(all_keys_0) > 0:
                store_db(database, idx, join(COMMON_DIR, f'{key_word1}_common.db'))
    # first = 0; second = 1
    # if (len(all_keys.keys()) > 1):
    #     if len(all_keys[0]) > len(all_keys[1]):
    #         first = 1; second = 0
    #     res, res_key = compare_points_and_keys(all_points, all_keys, first, second)    
    #     print(f'all points of db {first} exits in db {second}: {res}')
    #     print(f'all keys of db {first} exits in db {second}: {res_key}')
        
    #     for idx, p in all_points[first].items():
    #         if idx not in all_points[second]:
    #             print(f'found a point that does not exist in {second}')
    #             break
    #         else:
    #             if p != all_points[second][idx]:
    #                 print(f'not the same {p} vs {all_points[second][idx]}')
    #                 #break

def plot_bar_graphs(ax, ya, yb):
    """Plot two bar graphs side by side, with letters as x-tick labels."""
    x = np.arange(len(ya))
    width = 0.33
    ax.bar(x, ya, width)
    ax.bar(x + width, yb, width, color='C2')
    # ax.set_xticks(x + width, labels=['a', 'b', 'c', 'd', 'e'])
    return ax


if __name__ == '__main__':
    store = False
    store_csv = True
    read_csv = False
    if store:
        # FLAG = 'common-dse4'
        FLAG = 'common'
        for dataset in ['poly', 'machsuite']:
            db_path = f'{get_root_path()}/{dataset}/databases/**/*'
            if dataset == 'machsuite':
                COMMON_DIR = join(get_root_path(), f'machsuite/databases/{FLAG}-original-size/') ## extra: in v18 but not in v20
                KERNELS = MACHSUITE_KERNEL
            elif dataset == 'poly':
                COMMON_DIR = join(get_root_path(), f'poly/databases/{FLAG}/')
                KERNELS = poly_KERNEL
            create_dir_if_not_exists(COMMON_DIR)
            for kernel in KERNELS:
                # if kernel != 'aes':
                #     continue
                print(f'now processing {kernel}')
                compare_db_object(db_path, kernel , store_common=False, COMMON_DIR = COMMON_DIR)
                # break
            # break
        if store_csv:
            log_dict_of_dicts_to_csv(f'common_v18_v20', csv_dict, csv_dict['header'])
            save_pickle(csv_dict, 'common_v18_v20.pickle')
    elif read_csv:
        res = read_csv_dict(f'common_v18_v20', csv_dict, csv_dict['header'])                
    else:
        with open('common_v18_v20.pickle', 'rb') as f:
            all_common_data = pickle.load(f)

        print('total common:', len(all_common_data))
        common_timeout = 0
        one_timeout_1, one_timeout_2 = 0, 0
        remaining = 0
        remaining_values = {f'{ver}-{obj}': [] for ver in ['v18', 'v20'] for obj in objectives}
        for key, value in all_common_data.items():
            if key == 'header': continue
            for k in remaining_values:
                remaining_values[k].append(value[k])
        print('common timeout:', common_timeout, 'and one timeout:', one_timeout_1, 'and', one_timeout_2,
            'remaining:', remaining)

        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        points_dict = {obj: {'true': [], 'pred': []} for obj in objectives}
        for ind, obj in enumerate(objectives):
            # plot_bar_graphs(ax[ind], remaining_values[f'v18-{obj}'], remaining_values[f'v20-{obj}'])
            points_dict[obj]['pred'] = [(a,b) for a, b in zip(remaining_values[f'v18-{obj}'], remaining_values[f'v20-{obj}'])]
            points_dict[obj]['true'] = [(a,a) for a in remaining_values[f'v18-{obj}']]
            # ax[ind].legend(['v1', 'v2'])
        plot_points_with_subplot(points_dict, 'pred_true-v18_vs_v20_2obj', '', objectives)
        _report_rmse_etc(points_dict)
        # plt.title(obj)
        # plt.show()
        # plt.savefig('common_v18_v20.png', bbox_inches='tight')


    """
    # create a redis database
    database = redis.StrictRedis(host = 'localhost', port = 6379)
    
    # open a sample database
    f_db = open("./all_dbs/hog_result.db", "rb")
    
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    data = pickle.load(f_db)
    database.hmset(0, data)
    keys = [k.decode('utf-8') for k in database.hkeys(0)]
    # get a value of a specific key (key i)
    i = 6
    pickle_obj = database.hget(0, keys[i])
    obj = pickle.loads(pickle_obj)
    
    print(obj.ret_code, obj.perf, obj.res_util)  ## --> return code for the current configuration. checkout autodse/result.py for all available options
    # other variables that exists:
    ## obj.perf  --> the cycle counts of the given configuration
    ## obj.res_util --> the resource utilization for each type of the resourse (BRAM, DSP, LUT, FF)
    ##		    the variables "util_*" show the percentage of utilization
    ##		    the variables "total_*" show the exact number of resources used
    ## obj.eval_time --> how long it took for the tool to synthesize the given configuration
    ## obj.point --> the same as key. shows the value of each pragma
    ## obj.quality --> quality of the given design point as a measure of finite difference as defined in the paper 
    ##		   (ratio of difference of cycle and utilization compared to a base design)
    """
