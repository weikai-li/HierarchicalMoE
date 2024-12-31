from utils import get_root_path, MLP, print_stats
import os.path as osp
from os.path import join, basename
from glob import glob, iglob
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import networkx as nx
import redis, pickle, random
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack

from tqdm import tqdm

from shapely.geometry import Point, LineString

from utils import save_fig, create_dir_if_not_exists
import pickle
from copy import deepcopy

# import aug21autodse
# from localdse import result as l_result
# from autodse import result
# from rename_pickle import renamed_load, renamed_loads
import rename_pickle
# import autodse

class LineInterp(object):
    def __init__(self, x1, y1, x2, y2, num_points):
        assert (num_points >= 2)
        self.x1, self.y1, self.x2, self.y2, self.num_points = \
            x1, y1, x2, y2, num_points
    def name(self):
        return 'line_interp_x1={}_y1={}_x2={}_y2={}_{}ps'.format(
            self.x1, self.y1, self.x2, self.y2, self.num_points)

confs = [LineInterp(x1=-9.5, y1=-9, x2=-9, y2=-8, num_points=12)]
COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
PERF = 'perf'
perplexity = 25
n_iter = 5000

# MERGE_DIR = join(get_root_path(), 'machsuite/databases/v20-original-size/merged')
# MERGE_DIR = join(get_root_path(), 'machsuite/databases/original-size/single-merged')
# MERGE_DIR = join(get_root_path(), 'poly/databases/single-merged')
MERGE_DIR = join(get_root_path(), 'poly/databases/v20/merged')
create_dir_if_not_exists(MERGE_DIR)

### __PARA__L1-1.__PARA__L2-1.__PARA__L3-1.__PARA__L4-1.__PARA__L5-1.__PARA__L6-1.__PARA__L8-4.__PIPE__L0-NA.__PIPE__L1-NA.__PIPE__L10-NA.__PIPE__L11-off.__PIPE__L2-NA.__PIPE__L3-off.__PIPE__L4-flatten.__PIPE__L5-off.__PIPE__L6-flatten.__PIPE__L8-off.__TILE__L0-1.__TILE__L1-1.__TILE__L10-1.__TILE__L11-1.__TILE__L2-1.__TILE__L3-1.__TILE__L4-1.__TILE__L5-1.__TILE__L6-1.__TILE__L8-1


# DATASET = 'cnn1'
DATASET = 'machsuite'
if DATASET == 'cnn1':
    DIR = './save/cnn1'
    KERNEL = 'cnn'
    gexf_file = './new_db/case1/CnnKernel1.cpp.gexf'
    db_path = './databases/cnn_case1/'
elif DATASET == 'machsuite':
    DIR = './save/machsuite/gemm-ncubed'
    KERNEL = 'gemm-ncubed'
    gexf_file = './machsuite/dot-files/gemm-ncubed_kernel.cpp.gexf'
    # db_path = './machsuite/databases/**/*'
    db_path = './**/databases/**/*'

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['atax', 'mvt', 'gemver', 'gemm-p', '2mm', 'bicg', 'doitgen', 'gesummv']
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

# replace_from = 'flatten'
replace_from = 'flatten'

def write_db(database, KERNEL):
    db_path = join(MERGE_DIR, f'{KERNEL}_result_new_updated-0.db')
    dump_db = {
        key: database.hget(0, key)
        for key in database.hgetall(0)
    }
    with open(db_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)

def extract_db(KERNEL, combine = True):
    database = redis.StrictRedis(host='localhost', port=6379)
    new_database = redis.StrictRedis(host='localhost', port=8888)
    database.flushdb()
    new_database.flushdb()
    if DATASET == 'cnn1':
        db_files = [f for f in glob(join(db_path, '*.db'))]
    else:
        # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'v20' not in f and 'single-merged' in f] # and 'merged' not in f and 'transfer' in f]
        db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'v20' in f] # and 'merged' not in f and 'transfer' in f]
    print(db_files)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    idx_db = 0
    for idx, file in enumerate(db_files):
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        database.hmset(idx_db, data)
        keys = [k.decode('utf-8') for k in database.hkeys(idx_db)]
        print(file)
        for key in keys:
            pickle_obj = database.hget(idx_db, key)
            # print(key)
            # obj = pickle.loads(pickle_obj.replace(b'aug21autodse', b'autodse'))
            obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
            # obj = pickle.loads(pickle_obj)

            pickled_result = pickle.dumps(obj)
            new_database.hset(idx_db, key, pickled_result)


        #### remove unwanted keys:
        # keys = [k.decode('utf-8') for k in database.hkeys(idx_db)]
        # print(file)
        # for key in keys:
        #     pickle_obj = database.hget(idx_db, key)
        #     obj = pickle.loads(pickle_obj)
        #     if type(obj) is int or type(obj) is dict:
        #         continue
        #     if len(obj.point) != 5:
        #         database.hdel(idx_db, key)
        #         print(f'deleted')
        # keys = [k.decode('utf-8') for k in database.hkeys(idx_db)]

        # dump_db = {
        #     key: new_database.hget(idx_db, key)
        #     for key in new_database.hgetall(idx_db)
        # }
        # # with open(file.replace('.db-mod-mod', '.db'), 'wb') as filep:
        # with open(file, 'wb') as filep:
        #     pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)
        if not combine:
            idx_db += 1
        max_idx = idx + 1
    # keys = [k.decode('utf-8') for k in database.hkeys(0)]
    # print(keys[0])
    # print(f'number of keys: {len(keys)}')

    return keys, database


def process_db(keys):
    g = nx.read_gexf(gexf_file)   
    visited_config = set()
    groups_list = []
    groups_diff_list = []
    groups_categories = {}
    # sort by flatten pragma
    keys = sorted(keys)
    count = 0; count_flatten = 0; count_diff = 0
    for i in range(len(keys)):
        pickle_obj = database.hget(0, keys[i])
        obj = pickle.loads(pickle_obj)
        if type(obj) is int or type(obj) is dict or keys[i] in visited_config:
            continue
        
        if keys[i][0:3] == 'lv1':
            continue
        count += 1
        if keys[i] not in visited_config and replace_from in keys[i]:
            print(i, keys[i], obj.point)
            break
            filtered_configs = [k for k in obj.point if obj.point[k] == replace_from]
            visited_config.add(keys[i])
            count_flatten += 1
            #print(len(filtered_configs))
        
            for config in filtered_configs:
                group = []
                perfs = set()
                group.append(keys[i])
                perfs.add(obj.perf)
                for replace_to in ['NA', 'off']:
                    new_key = keys[i].replace(config+'-'+replace_from, config+'-'+replace_to)
                    visited_config.add(new_key)
                    #print(config)
                    #print(new_key)
                    #print(database.exists(0, new_key))
                    if database.hexists(0, new_key):
                        group.append(new_key)
                        visited_config.add(new_key)
                        new_pickle_obj = database.hget(0, new_key)
                        new_obj = pickle.loads(new_pickle_obj)
                        perfs.add(new_obj.perf)
                        #print(obj.perf, new_obj.perf)
                if len(group) > 2:
                    groups_list.append(group)
                    if config in groups_categories:
                        g_list = groups_categories[config]
                    else:
                        g_list = []
                    g_list.append(group)
                    groups_categories[config] = g_list
                    
                    if len(perfs) > 0 and not (len(perfs) == 1 and 0.0 in perfs):
                        count_diff += 1
                        groups_diff_list.append((config, group))


    print(f' number of configs: {count} \n number of flatten pragams: {count_flatten} \n number of pipeline groups: \
        {len(groups_list)} \n number of different performances in pipeline pragmas: {count_diff}')
    
    # for g in groups_list:
    #     for gg in g:
    #         print(gg)
    #     print()

    perfs_set = set()
    for group_key, group in groups_diff_list:
        for g in group:
            new_pickle_obj = database.hget(0, g)
            new_obj = pickle.loads(new_pickle_obj)
            perfs_set.add(new_obj.perf)
    # for group_key, group in groups_diff_list:
    #     perfs = []
    #     for g in group:
    #         new_pickle_obj = database.hget(0, g)
    #         new_obj = pickle.loads(new_pickle_obj)
    #         perfs.append(new_obj.perf)
    #     perf_list = [p / perfs[2] for p in perfs]
    #     plt.plot(perf_list)
    #     print(group_key, perf_list)
        
    fig_id = 0
    for group_key, group_list in groups_categories.items():
        plt.figure(fig_id)
        for group in group_list:
            perfs = []
            for g in group:
                new_pickle_obj = database.hget(0, g)
                new_obj = pickle.loads(new_pickle_obj)
                perfs.append(new_obj.perf)
            perf_list = [p / max(perfs_set) for p in perfs]
            plt.plot(perf_list)
            print(group_key, perf_list)
        
        plt.title(f'{group_key}')
        plt.xticks(np.arange(0, 1, step=0.2))  # Set label locations.
        plt.xticks(np.arange(3), [replace_from, 'NA', 'off'])  # Set text labels.    
        if count_flatten > 0:
            save_fig(plt, join(DIR, 'pipeline-plots'), f'{fig_id}-{group_key}-different-perfs-pipeline-pragma-flatten-NA-off-normalized-to-max-value', print_path=True)
        fig_id += 1
    plt.close()


if __name__ == '__main__':
    if DATASET == 'cnn1':
        keys, database = extract_db('cnn1')
        process_db(keys)
    else:
        # for k in ALL_KERNEL:
        for k in poly_KERNEL:
            # if k == 'atax':
            print(k)
            keys, database = extract_db(k, combine=True)
            write_db(database, k)
            # break

