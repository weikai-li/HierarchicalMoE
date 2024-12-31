import redis
import pickle5 as pickle
from collections import Counter
from os.path import join, basename, dirname
from glob import glob, iglob
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
from copy import deepcopy

from utils import get_root_path, create_dir_if_not_exists
from result import Result
import pickletools
from autodse.result import MerlinResult


# COMMON_DIR = join(get_root_path(), 'machsuite/databases/common-original-size/')
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
                    'nw', 'md', 'stencil-3d']
# MACHSUITE_KERNEL = []

poly_KERNEL = [
    # 'doitgen-red',
    # 'trmm',
    # 'bicg-large', 'covariance', 'gemm-p-large', 'fdtd-2d-large',
    # 'trmm-opt', 'trmm', 'symm-opt', 'syrk',
    # 'symm',
    # 'syr2k', 'correlation'
    # 'mvt-medium'
    'atax-medium'
    ]

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium',
               'gemver-medium']

VER = 'v18'
VER = 'v20'
VER = 'v21'



def init_db(port=6379):
    database = redis.StrictRedis(host='localhost', port=port)
    database2 = redis.StrictRedis(host='localhost', port=7777)
    database.flushdb()
    database2.flushdb()
    return database, database2

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
    
    
def replace_module_name(pickled_data, old_module_name=b'localdse', new_module_name=b'autodse'):
    # Find occurrences of the old module name
    index = pickled_data.find(old_module_name)
    

    while index != -1:
        # Use pickletools to find the opcode for the NEWOBJ oparg
        opcodes = list(pickletools.genops(pickled_data))
        newobj_index = next((i for i, op in enumerate(opcodes) if type(op[0]) == str and b'localdse' in op[0]), -1)

        # Calculate the position to replace the old module name with the new one
        if newobj_index != -1:
            position = opcodes[newobj_index][1][0]
            pickled_data = pickled_data[:position] + new_module_name + pickled_data[position + len(old_module_name):]
        print(f'{opcodes=}')
        print(f'{newobj_index=}')
        # Find the next occurrence
        index = pickled_data.find(old_module_name, index + len(new_module_name))

    return pickled_data
    
def update_db_name(pickled_obj):
    # print(f'{pickled_obj=}')
    # print(f'{replace_module_name(pickled_obj)=}')
    # new_obj = pickle.loads(replace_module_name(pickled_obj))
    new_obj = pickle.loads(pickled_obj.replace(b'autudse', b'autodse'))
    # new_obj = MerlinResult()
    # if hasattr(obj, 'ordered_paths'):
    #     print(f'{obj.ordered_paths=}')
    
    # for attr_name in vars(obj):
    #     if attr_name == 'ret_code':
    #         obj_subclass = getattr(obj, attr_name)
    #         print(obj_subclass.__dict__)
    #         # Deep copy attributes from instance_a to instance_b
    #         copied_attributes = deepcopy(obj_subclass.__dict__)
    #         copied_obj = new_obj.ret_code
    #         # Update instance_b with copied attributes
    #         copied_obj.__dict__.update(copied_attributes)
    #         setattr(new_obj, attr_name, copied_obj)
    #     else:
    #         setattr(new_obj, attr_name, getattr(obj, attr_name))
    
    return new_obj
    
def get_db_files(db_path, key_word1 = None, FLAG = 'one-db'):
    if FLAG == 'one-db':
        if gather:
            db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 'test' not in f and 'one-db' not in f and 'corrupt' not in f and 'wrong' not in f and 'task-transfer' not in f and ('iccad' in f and VER in f)] ## poly v18 
        else:
            db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 'test' not in f and 'corrupt' not in f and 'wrong' not in f and 'archive' in f and VER in f and 'single-merged' not in f] # and ('round12' in f or 'round13' in f)] 
    elif FLAG == 'extra':
        db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 'test' not in f and 'corrupt' not in f and 'wrong' not in f and 'archive' not in f and 'task' not in f and (f'one-db-extended-round{round_num}' in f)] 
        assert len(db_files) == 1
        db_files.extend([f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 'test' not in f and 'corrupt' not in f and 'wrong' not in f and 'archive' not in f and 'task' not in f and f'one-db-extended-round{round_num-1}' in f]) # and 'norm-perf' in f ##   poly v18
        assert len(db_files) == 2
    elif FLAG == 'common': ## common between v20 and v18
        db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 'v20' in f and 'test' not in f and 'corrupt' not in f and 'wrong' not in f and 'archive' not in f and 'task' not in f and 'extended' not in f] 
        if len(db_files) != 1:
            return None ## not passed the db criteria
        assert len(db_files) == 1
        db_files.extend([f for f in iglob(join(db_path, '*.db'), recursive=True) if f'{key_word1}_' in f and 'v18' in f and 'test' not in f and 'corrupt' not in f and 'wrong' not in f and 'archive' not in f and 'task' not in f and f and 'extended' not in f and 'large' not in f]) # and 'norm-perf' in f ##   poly v18
        print(len(db_files))
        assert len(db_files) == 2
    return db_files

def add_if_not_exists_or_now_valid(database, database_new, len_pragma = 0, idx_db = 0):
    keys_new = [k.decode('utf-8') for k in database_new.hkeys(idx_db)]
    for key in sorted(keys_new):
        if key == 'lv1:' or key == 'lv2:':
            database.hdel(idx_db, key)
            continue
        if database.hexists(idx_db, key):
            pickled_obj_new = database_new.hget(idx_db, key)
            pickled_obj = database.hget(idx_db, key)
            obj_new = update_db_name(pickled_obj_new)
            obj = update_db_name(pickled_obj)
            # obj_new = pickle.loads(pickled_obj_new.replace(b'localdse', b'autodse'))
            # obj = pickle.loads(pickled_obj.replace(b'localdse', b'autodse'))
            
            if type(obj_new) is int or type(obj_new) is dict:
                if type(obj_new) is dict:
                    if key == 'scope-map':
                        continue
                    assert obj == obj_new, f'{obj}, {obj_new}'
                elif obj_new > obj:
                    database.hset(idx_db, key, pickle.dumps(obj_new))
                continue
            # if len(obj.point) == 21:
            #     print('not adding key', key)
            #     continue
            # print(key)
            if 'lv' in key: assert len(obj_new.point) == len_pragma, print(len(obj.point), len_pragma)
            if obj_new.ret_code == 'EARLY_REJECT':
                obj_new.ret_code = Result.RetCode.EARLY_REJECT
            if obj.ret_code.name == obj_new.ret_code.name and obj.perf == obj_new.perf:
                continue
            elif obj.ret_code.name == obj_new.ret_code.name and obj.perf != obj_new.perf:
                print(f'invalid design config with key: {key}')
                print(f'the prev obj had perf {obj.perf} with ret code: {obj.ret_code.name}')
                print(f'but now has perf {obj_new.perf} with ret code: {obj_new.ret_code.name}')
                print()
                raise RuntimeError()
            elif obj_new.ret_code.name == 'PASS' or obj_new.ret_code.name == 'EARLY_REJECT':
                if pickled_obj_new:
                    database.hset(idx_db, key, pickle.dumps(obj_new))
                if obj.ret_code.name == 'TIMEOUT':
                    continue
                print(f'for {key}')
                print(f'replaced prev point with perf {obj.perf} and ret code: {obj.ret_code.name}')
                print(f'with perf {obj_new.perf} with ret code: {obj_new.ret_code.name}')
                print()
            else:
                if obj_new.ret_code.name == 'TIMEOUT' and (obj.ret_code.name == 'PASS' or obj.ret_code.name == 'EARLY_REJECT'):
                    continue
                print(f'for {key}')
                print(f'ignored new point with perf {obj_new.perf} with ret code: {obj_new.ret_code.name}')
                print(f'prev point was with perf {obj.perf} and ret code: {obj.ret_code.name}')
                print()
        else:
            pickled_obj = database_new.hget(idx_db, key)
            obj = update_db_name(pickled_obj)
            # obj = pickle.loads(pickled_obj.replace(b'localdse', b'autodse'))
            # if len(obj.point) == 21:
            #     print('not adding key', key)
            #     continue
            if 'lv' in key: assert len(obj.point) == len_pragma, print(len(obj.point), len_pragma)
            if pickled_obj:
                if type(obj) is int or type(obj) is dict:
                    pass
                elif obj.ret_code == 'EARLY_REJECT':
                    obj.ret_code = Result.RetCode.EARLY_REJECT
                database.hset(idx_db, key, pickle.dumps(obj))
    setup = database.hget(idx_db, 'setup')
    if setup: obj_tool = pickle.loads(setup)
    else: 
        if VER == 'v18': database.hset(0, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
        elif VER == 'v20': database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
        elif VER == 'v21': database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-21.1'}))
        else: raise NotImplementedError()
    # print(ast.literal_eval(obj_tool))
    # print(obj_tool)
    if setup:
        if VER == 'v18': assert obj_tool['tool_version']== 'SDx-18.3'
        elif VER == 'v20': assert obj_tool['tool_version']== 'Vitis-20.2'
        elif VER == 'v21': assert obj_tool['tool_version']== 'Vitis-21.1'
        else: raise NotImplementedError()

def extract_new_or_common_config(database, database_prev, len_pragma = 0, idx_db_new = 1, FLAG = None):
    idx_db = 0
    keys_new = [k.decode('utf-8') for k in database.hkeys(idx_db)]
    for key in sorted(keys_new):
        if key == 'lv1:' or key == 'lv2:':
            database.hdel(idx_db, key)
            continue
        cond = database_prev.hexists(idx_db, key) ## for 'extra'
        if FLAG == 'common': cond = not cond
        if cond:
            pass
        else:
            pickled_obj = database.hget(idx_db, key)
            obj = update_db_name(pickled_obj)
            # obj = pickle.loads(pickled_obj.replace(b'localdse', b'autodse'))
            if 'lv' in key: assert len(obj.point) == len_pragma, print(len(obj.point), len_pragma)
            if pickled_obj:
                if type(obj) is int or type(obj) is dict:
                    pass
                elif obj.ret_code == 'EARLY_REJECT':
                    obj.ret_code = Result.RetCode.EARLY_REJECT
                database.hset(idx_db_new, key, pickle.dumps(obj))
    setup = database.hget(idx_db, 'setup')
    if setup: obj_tool = pickle.loads(setup)
    else: 
        if VER == 'v18': database.hset(idx_db_new, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
        elif VER == 'v20': database.hset(idx_db_new, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
        elif VER == 'v21': database.hset(idx_db_new, 'setup', pickle.dumps({'tool_version': 'Vitis-21.2'}))
        else: raise NotImplementedError()
    # print(ast.literal_eval(obj_tool))
    # print(obj_tool)
    if setup:
        if VER == 'v18': assert obj_tool['tool_version']== 'SDx-18.3'
        elif VER == 'v20': assert obj_tool['tool_version']== 'Vitis-20.2'
        elif VER == 'v21': assert obj_tool['tool_version']== 'Vitis-21.2'
        else: raise NotImplementedError()

def verify_db(database, idx_db = 0):
    keys_new = [k.decode('utf-8') for k in database.hkeys(idx_db)]
    for key in sorted(keys_new):
        pickled_obj = database.hget(idx_db, key)
        obj = pickle.loads(pickled_obj)
        if type(obj) is int or type(obj) is dict:
            continue
        obj = update_db_name(pickled_obj)
        if obj.ret_code == 'EARLY_REJECT':
            obj.ret_code = Result.RetCode.EARLY_REJECT
            # database.hset(idx_db, key, pickle.dumps(obj))
            
        database.hset(idx_db, key, pickle.dumps(obj))

def read_db(database, database2, db_files, combine = True, idx_db = 0, FLAG = 'one-db'):
    '''
    combine: merge all the databases as one or read them separately
    '''
    # idx_db = 0
    len_pragma = 0
    for idx, file in enumerate(db_files):
        print('###########')
        print(file)
        print('###########')
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        if idx == 0:
            database.hset(idx_db, mapping=data)
            verify_db(database, idx_db)
            keys = [k.decode('utf-8') for k in database.hkeys(idx) if 'lv2' in k.decode('utf-8')]
            obj = pickle.loads(database.hget(idx, keys[0]))
            len_pragma = len(obj.point)
        else:
            database2.flushdb()
            database2.hset(idx_db, mapping=data)
            verify_db(database2, idx_db)
            if FLAG == 'one-db':
                add_if_not_exists_or_now_valid(database, database2, len_pragma, idx_db)
            elif FLAG == 'extra' or FLAG == 'common':
                extract_new_or_common_config(database, database2, len_pragma, idx_db_new=1, FLAG=FLAG) ## the new config will be in idx_db=1 of database
            else:
                raise NotImplementedError()
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
        

# run "redis-server" on command line first!
def merge_db_object(db_path, key_word1 = None, normalize_perf=False, store_common=True, COMMON_DIR = None, FLAG = 'one-db'):
    # create a redis database
    database, database2 = init_db()

    ## read v18
    db_files = get_db_files(db_path, key_word1, FLAG)
    if db_files == None:
        return ## no db was found for the given test
    if FLAG == 'one-db': db_files = sorted(db_files)
    pprint(db_files)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    num_db = read_db(database, database2, db_files, combine = True, idx_db = 0, FLAG = FLAG)
    idx_db = 0
    if FLAG == 'extra' or FLAG == 'common':
        idx_db = 1
    all_keys_0 = get_keys_with_id(database, idx_db)
    valid_keys_0 = [k for k in all_keys_0 if 'lv2' in k]
    print(f'keys1: {len(all_keys_0)}')
    print(f'valid keys1: {len(valid_keys_0)}')

    if len(all_keys_0) > 0:
        store_db(database, idx_db, join(COMMON_DIR, f'{key_word1}_result_updated-0.db'))
    


def plot_bar_graphs(ax, ya, yb):
    """Plot two bar graphs side by side, with letters as x-tick labels."""
    x = np.arange(len(ya))
    width = 0.33
    ax.bar(x, ya, width)
    ax.bar(x + width, yb, width, color='C2')
    # ax.set_xticks(x + width, labels=['a', 'b', 'c', 'd', 'e'])
    return ax


if __name__ == '__main__':
    store = True
    if store:
        FLAG = 'one-db'
        # FLAG = 'extra'
        # FLAG = 'common'
        round_num = 1 if VER == 'v21' else 3 if VER == 'v20' else 13
        gather = False
        if FLAG == 'extra' or FLAG == 'common':
            gather = False
        
        target = ['machsuite', 'poly']
        # target = ['poly']
        # target = ['machsuite']
        for dataset in target:
            # db_path = f'{dirname(get_root_path())}/src/logs/auto-encoder/extended-graph-db/round2/**/*'
            if gather:
                db_path = f'{dirname(get_root_path())}/src/logs/auto-encoder/**/*' ## ' norm-perf in it
                # db_path = f'{dirname(get_root_path())}/src/logs/**/*' ## ' norm-perf in it
            else:
                db_path = f'{get_root_path()}/{dataset}/databases/**/*'
        # for dataset in ['poly']:
            if dataset == 'machsuite':
                # COMMON_DIR = join(get_root_path(), 'machsuite/databases/original-size/single-merged/') ## extra: in v18 but not in v20
                # COMMON_DIR = join(get_root_path(), f'machsuite/databases/{VER}/original-size/one-db-extended-round9/') 
                COMMON_DIR = join(get_root_path(), f'machsuite/databases/{VER}/original-size/{FLAG}-extended-round{round_num}/') 
                # COMMON_DIR = join(get_root_path(), f'machsuite/databases/{VER}/original-size/archive/single-merged') 
                KERNELS = MACHSUITE_KERNEL
            elif dataset == 'poly':
                COMMON_DIR = join(get_root_path(), f'poly/databases/{VER}/one-db/')
                COMMON_DIR = join(get_root_path(), f'poly/databases/{VER}/one-db/{FLAG}-extended-round{round_num}')
                # COMMON_DIR = join(get_root_path(), f'poly/databases/{VER}/archive/single-merged')
                KERNELS = poly_KERNEL
            create_dir_if_not_exists(COMMON_DIR)
            for kernel in KERNELS:
                if True or 'md' in kernel:
                    print(f'now processing {kernel}')
                    merge_db_object(db_path, kernel , store_common=True, COMMON_DIR = COMMON_DIR, FLAG = FLAG)

                    print('***********************************************')
                #break
            # break
   

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
