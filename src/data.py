from config import FLAGS
import config
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL
from saver import saver
from utils import get_root_path, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load
from result import Result
from coreset import kmeans_split

import os
import json
from os.path import join, basename
from glob import glob, iglob
import re

from math import ceil

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data

import networkx as nx
import redis 
# import pickle5 as pickle    # pickle5 only supports python <= 3.8
import pickle
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack, coo_matrix

from tqdm import tqdm

import os.path as osp

import torch
from torch_geometric.data import Dataset
from torch.utils.data import random_split

from shutil import rmtree
import math
import random


NON_OPT_PRAGMAS = ['LOOP_TRIPCOUNT', 'INTERFACE', 'INTERFACE', 'KERNEL']
WITH_VAR_PRAGMAS = ['DEPENDENCE', 'RESOURCE', 'STREAM', 'ARRAY_PARTITION']
#SAVE_DIR = join(get_save_path(), FLAGS.dataset, f'run10-with_kernel_name-no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}')
TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
# SAVE_DIR = join(get_save_path(), FLAGS.dataset, f'with-updated-up-tile-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}')
## regressions for FPGA'22 and DAC'22
# SAVE_DIR = join(get_save_path(), FLAGS.dataset, f'with-updated-up5-tile-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}')
# SAVE_DIR = join(get_save_path(), FLAGS.dataset, f'with-updated-up3-tile-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}')
# SAVE_DIR = join(get_save_path(), FLAGS.dataset, f'pragma-with-updated-up4-tile-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}')
# SAVE_DIR = join(get_save_path(), FLAGS.dataset, f'smaller_normalizer-{FLAGS.task}_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(FLAGS.target)}')

if FLAGS.data_loading_path != '':
    SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', FLAGS.data_loading_path)
else:
    if ALL_KERNEL == ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
            'nw', 'md', 'stencil-3d', '2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large',
            'covariance', 'doitgen', 'doitgen-red', 'fdtd-2d', 'gemm-p-large', 'gemver', 'gesummv',
            'heat-3d', 'jacobi-1d', 'mvt', 'seidel-2d', 'symm', 'symm-opt', 'syrk', 'trmm',
            'mvt-medium', 'correlation', 'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium']:
        SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', "pretrain_dataset")
    elif ALL_KERNEL == ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
            'nw', 'md', 'stencil-3d', '2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance',
            'doitgen', 'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large',
            'gemver', 'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
            'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
            'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium', 'gemver-medium']:
        SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', "whole_dataset")
    elif ALL_KERNEL == ['fdtd-2d-large', 'gemver-medium', 'syr2k', 'gemm-p', 'jacobi-2d', 'trmm-opt']:
        SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', "finetune_dataset")
    elif ALL_KERNEL == ['3d-rendering', 'att-3mm', 'att-3mm-fuse', 'spam-filter', 'vmmv']:
        SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', "complex_kernels")
    elif len(ALL_KERNEL) == 1:
        SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', ALL_KERNEL[0])
    else:
        raise NotImplementedError()

ENCODER_PATH = join(SAVE_DIR, 'encoders')
# PROCESSED_DIR = join(SAVE_DIR, 'processed')
create_dir_if_not_exists(SAVE_DIR)

print(SAVE_DIR)

# DATASET = 'cnn1'
DATASET = 'machsuite-poly'
if DATASET == 'cnn1':
    KERNEL = 'cnn'
    db_path = '../dse_database/databases/cnn_case1/'
elif DATASET == 'machsuite':
    KERNEL = FLAGS.tag
    db_path = '../dse_database/machsuite/databases/**/*'
elif DATASET == 'machsuite-poly':
    KERNEL = FLAGS.tag
    db_path = []
    for benchmark in FLAGS.benchmarks:
        db_path.append(f'../dse_database/{benchmark}/databases/**/*')
    

is_normal_gexf_folder = True
if FLAGS.dataset == 'vitis-cnn':
    GEXF_FOLDER = join(get_root_path(), 'dse_database', 'dotGenerator_all_kernels')
elif FLAGS.dataset == 'machsuite':
    GEXF_FOLDER = join(get_root_path(), 'dse_database', 'machsuite', 'dot-files')
elif FLAGS.dataset == 'programl':
    if FLAGS.data_loading_path == '' or 'harp_graph' in FLAGS.data_loading_path:
        GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')
    else:
        assert 'raw_graph' in FLAGS.data_loading_path
        GEXF_FOLDER = join('../save/programl/contest/raw_graphs/*')
        is_normal_gexf_folder = False
        GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and 'processed' in f])
elif FLAGS.dataset == 'programl-machsuite':
    GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', 'machsuite', 'processed')
elif FLAGS.dataset == 'simple-programl':
    GEXF_FOLDER = join(get_root_path(), 'dse_database', 'simple-program', 'programl', 'processed', '**')
else:
    raise NotImplementedError()

    # GEXF_FILES = [f for f in sorted(glob(join(GEXF_FOLDER, '*.gexf'))) if f.endswith('.gexf') and KERNEL in f]

if is_normal_gexf_folder:
    if FLAGS.all_kernels:
        if FLAGS.graph_type == '':
            print('data.py Line 116 wrong! exit!')
            exit()
            GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and 'extended' not in f and 'processed' in f])
        else:
            if 'hierarchy' not in FLAGS.graph_type:
                pruner = 'hierarchy'
            else:   # This is our case
                pruner = 'initial'
            if FLAGS.use_for_nodes:
                GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and FLAGS.graph_type in f and 'processed' in f and pruner not in f and 'pseudo-for' in f])
            else:
                GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and FLAGS.graph_type in f and 'processed' in f and pruner not in f and 'pseudo-for' not in f])
    else:
        if FLAGS.use_for_nodes:
            GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and f'{FLAGS.target_kernel}_' in f and 'extended' not in f and 'processed' in f and 'pseudo-for' in f])
        else:
            GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and f'{FLAGS.target_kernel}_' in f and 'extended' not in f and 'processed' in f and 'pseudo-for' not in f])


def finte_diff_as_quality(new_result: Result, ref_result: Result) -> float:
    """Compute the quality of the point by finite difference method.

    Args:
        new_result: The new result to be qualified.
        ref_result: The reference result.

    Returns:
        The quality value (negative finite differnece). Larger the better.
    """

    def quantify_util(result: Result) -> float:
        """Quantify the resource utilization to a float number.

        util' = 5 * ceil(util / 5) for each util,
        area = sum(2^1(1/(1-util))) for each util'

        Args:
            result: The evaluation result.

        Returns:
            The quantified area value with the range (2*N) to infinite,
            where N is # of resources.
        """

        # Reduce the sensitivity to (100 / 5) = 20 intervals
        try:
            utils = [
                5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon for k, u in result.res_util.items()
                if k.startswith('util')
            ]
        except:
            utils = [
                5 * ceil(u * 100 / 5) / 100 + FLAGS.epsilon for k, u in result.res_util.items()
                if k.startswith('util')
            ]

        # Compute the area
        return sum([2.0**(1.0 / (1.0 - u)) for u in utils])

    ref_util = quantify_util(ref_result)
    new_util = quantify_util(new_result)

    # if (new_result.perf / ref_result.perf) > 1.05:
    #     # Performance is too worse to be considered
    #     return -float('inf')

    if new_util == ref_util:
        if new_result.perf < ref_result.perf:
            # Free lunch
            # return float('inf')
            return FLAGS.max_number
        # Same util but slightly worse performance, neutral
        return 0

    return -(new_result.perf - ref_result.perf) / (new_util - ref_util)
    

class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None, e_kernel=None):
        # self.processed_dir = PROCESSED_DIR
        global SAVE_DIR
        if e_kernel is not None:
            SAVE_DIR = '/'.join(SAVE_DIR.split('/')[:-1]) + f'/{e_kernel}'
        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)
        if data_files is not None:
            self.data_files = data_files
        self.file_list = self.processed_file_names

        # if FLAGS.train_mode not in ['maml']:
        #     for i in range(len(self.file_list)):
        #         data = torch.load(self.file_list[i])
        #         kernel_name = data['kernel'][:-17]
        #         try:
        #             assert kernel_name in ALL_KERNEL or kernel_name in ['stencil_stencil2d']
        #         except:
        #             print(f'{kernel_name} should not occur! You need to force_regen the data')
        #             exit()

        if FLAGS.load_data_to_device and data_files is not None and FLAGS.train_mode != 'maml':
            new_data_files = []
            for df in data_files:
                new_data = torch.load(df).to(FLAGS.device)
                new_data_files.append(new_data)
            data_files = new_data_files
            self.data_files = data_files

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        # return ['data_1.pt', 'data_2.pt', ...]
        if hasattr(self, 'data_files'):
            return self.data_files
        else:
            rtn = glob(join(SAVE_DIR, '*.pt'))
            return rtn

    def download(self):
        pass

    # Download to `self.raw_dir`.

    def process(self):
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()
    
    def get_file_path(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = self.file_list[idx]
        return fn

    def get(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = self.file_list[idx]
        if FLAGS.load_data_to_device == False or hasattr(self, 'data_files') == False or FLAGS.train_mode == 'maml':
            try:
                data = torch.load(fn)
            except:
                fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(2353))  # For the v21_granular finetune data
                data = torch.load(fn)
        else:
            data = self.data_files[idx]
        return data


def split_dataset(dataset, train, val, dataset_test=None, pragma_dim=None, current_train=[],
        num_features=None, edge_dim=None):
    file_li = dataset.processed_file_names
    kernel_lengths = []
    if FLAGS.finetune:
        if FLAGS.transfer_k_shot == 0:
            li = [[], [], file_li]
        else:
            groups = {}
            for i, fi in enumerate(file_li):
                a = torch.load(fi)
                kernel = a['gname']
                try:
                    groups[kernel].append(fi)
                except:
                    groups[kernel] = [fi]
            li = [[], [], []]

            for kernel, group in groups.items():
                if train < 1:
                    train_len = round(train * len(group))
                else:
                    train_len = train
                if len(group) <= train_len:
                    continue
                if FLAGS.coreset == 'random':
                    tmp_li = random_split(group, [train_len, 0, len(group) - train_len],
                            generator=torch.Generator().manual_seed(FLAGS.random_seed))
                elif FLAGS.coreset == 'kmeans':
                    tmp_li, selected_indices = kmeans_split(group,
                        lengths=[train_len, 0, len(group) - train_len], pragma_dim=pragma_dim)
                li[0].extend(tmp_li[0])
                li[1].extend(tmp_li[1])
                li[2].extend(tmp_li[2])
                kernel_lengths.append(len(group))
            print(len(li[0]), len(li[1]), len(li[2]), len(file_li))

    elif FLAGS.finetune == False:
        if train < 1 and val < 1:
            _m_test = 1 - train - val
        else:
            _m_test = len(dataset) - train - val
        li = random_split(file_li, [train, val, _m_test],
                          generator=torch.Generator().manual_seed(FLAGS.random_seed))

    if dataset_test is None:
        dataset_test = li[2]
    saver.log_info(f'{len(li[0]) + len(li[1]) + len(dataset_test)} graphs in total:'
          f' {len(li[0])} train {len(li[1])} val '
          f'{len(dataset_test)} test')
    train_dataset = MyOwnDataset(data_files=li[0])
    val_dataset = MyOwnDataset(data_files=li[1])
    test_dataset = MyOwnDataset(data_files=dataset_test)
    
    if FLAGS.finetune == True:
        return [train_dataset, val_dataset, test_dataset], li, kernel_lengths
    return [train_dataset, val_dataset, test_dataset]


def split_dataset_resample(dataset, train, val, test, test_id=0):
    file_li = dataset.processed_file_names
    # file_li = ['xxxx'] * len(file_li)
    num_batch = int(1 / test)
    splits_ratio = [int(len(dataset) * test)] * num_batch
    splits_ratio[-1] = len(dataset) - int(len(dataset) * test * (num_batch-1))
    print(splits_ratio, len(dataset), sum(splits_ratio))
    splits_ = random_split(file_li, splits_ratio,
                          generator=torch.Generator().manual_seed(100))
    test_split = splits_[test_id]
    train_val_data = []
    for i in range(num_batch):
        if i != test_id:
            train_val_data.extend(splits_[i])
    new_train, new_val = int(len(train_val_data) * train / (train+val)), len(train_val_data) - int(len(train_val_data) * train / (train+val))
    li = random_split(train_val_data, [new_train, new_val],
                          generator=torch.Generator().manual_seed(100))
    saver.log_info(f'{len(file_li)} graphs in total:'
          f' {len(li[0])} train {len(li[1])} val '
          f'{len(test_split)} test')
    train_dataset = MyOwnDataset(data_files=li[0])
    val_dataset = MyOwnDataset(data_files=li[1])
    test_dataset = MyOwnDataset(data_files=test_split)
    exit()
    # all_data = []
    # all_data.extend(li[0])
    # all_data.extend(li[1])
    # all_data.extend(test_split)
    # all_data = [int(((f.split('/')[-1]).split('.')[0]).split('_')[-1]) for f in all_data]
    # print(sorted(all_data))
    # print(len(all_data), sorted(all_data)[0], sorted(all_data)[-1])
    return train_dataset, val_dataset, test_dataset


def get_kernel_samples(dataset):
    samples = defaultdict(list)
    for data in dataset:
        if f'{FLAGS.target_kernel}_' in data.gname:
            samples[FLAGS.target_kernel].append(data)

    return samples[FLAGS.target_kernel]

def split_train_test_kernel(dataset):
    samples = defaultdict(list)
    assert FLAGS.test_kernels is not None, 'No test_kernels selected'
    print("splitting train and test kernels")
    for idx, data in tqdm(enumerate(dataset)):
        if any(f'{kernel_name}_' in data.kernel for kernel_name in FLAGS.test_kernels):
            samples['test'].append(dataset.get_file_path(idx))
        else:
            samples['train'].append(dataset.get_file_path(idx))

            
    data_dict = defaultdict()
    data_dict['train'] = MyOwnDataset(data_files=samples['train'])
    # data_dict['test'] = MyOwnDataset(data_files=samples['test'])
    data_dict['test'] = samples['test']

    return data_dict


def contest_get_data_list():
    #S0 Log the number of gexf files found in the specified folder
    #M0 directed graph files to right directory
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')#TODO: 1 graph files 
    
    #S1 create a redis database MD: no redis needed
    # database = redis.StrictRedis(host='localhost', port=6379)
    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    if FLAGS.data_loading_path[:11] == 'contest_v18':
        directory_path = '../save/programl/contest/train_designs/v18'
    elif FLAGS.data_loading_path[:11] == 'contest_v20':
        directory_path = '../save/programl/contest/train_designs/v20'
    elif FLAGS.data_loading_path[:11] == 'contest_v21':
        assert ('train' in FLAGS.data_loading_path and 'test' in FLAGS.data_loading_path) == False
        if 'train' in FLAGS.data_loading_path:
            directory_path = '../save/programl/contest/train_designs/v21'
        else:
            assert 'test' in FLAGS.data_loading_path
            directory_path = '../save/programl/contest/test_designs_v21'
    else:
        raise NotImplementedError()
    all_files = glob(f'{directory_path}/**/*', recursive=True)
    db_path = [file for file in all_files if not file.endswith('/')]
    
    #S2 choose encoder
    if FLAGS.encoder_path != None:
         # Load existing encoders
        saver.info(f'loading encoder from {FLAGS.encoder_path}')
        encoders = load(FLAGS.encoder_path, saver.logdir)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']
        
        enc_ftype_edge = encoders['enc_ftype_edge']
        enc_ptype_edge = encoders['enc_ptype_edge']

    else:
        # Create new OneHotEncoder objects for different types (handling unknown values)
        ## handle_unknown='ignore' is crucial for handling unknown variables of new kernels
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')
        
        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
    #S3: Initialize data structures for graph processing
    data_list = []

    all_gs = OrderedDict()

    # S3.1: Initialize lists to store encoded node and edge types
    X_ntype_all = []
    X_ptype_all = []
    X_itype_all = []
    X_ftype_all = []
    X_btype_all = []
    
    edge_ftype_all = []
    edge_ptype_all = []
    # S3.2: Other initialization for graph processing
    tot_configs = 0
    num_files = 0
    init_feat_dict = {}
    max_pragma_length = 21

    # S4: Process each graph file
    for gexf_file in tqdm(GEXF_FILES[0:]):
        saver.info(f'Working on graph file: {gexf_file}')
        # S4.1: Check the dataset and filter based on kernel names
        proceed = False
        for k in ALL_KERNEL:#TODO: make sure all com are in ALL_KERNEL
            if f'{k}_processed_result.gexf' in gexf_file:
                proceed = True
                break
        if not proceed:
            saver.info(f'Skipping this file as the kernel name is not selected. Check config file.')
            continue
        # pass
        
        
        # S4.2: Load the graph from the gexf file
        g = nx.read_gexf(gexf_file)
        g.variants = OrderedDict()
        
        gname = basename(gexf_file).split('.')[0]
        n = f"{basename(gexf_file).split('_')[0]}.json"
        #n = basename(gexf_file).split('_')[0] #Mn:
        
        saver.log_info(gname)
        all_gs[gname] = g
        
        # S4.3 Load the Redis database for the current graph
        # db_path: reddis /home/jade/HARP/dse_database/poly/databases/v21
        # M2:  change read file type
        db_paths = [db_p for db_p in db_path if db_p.endswith('.json') and n in db_p]
        assert len(db_paths) == 1
        
        # S4.4: M3 delete redis database as not needed 
        # saver.log_info(f'db_paths for {n}:')
        
        # for d in db_paths:
        #     saver.log_info(f'{d}')
        
        # S4.5: Load the database entries into Redis
        #print("\n Finish Data Path work 4.5 \n")
        
        # load the database and get the keys & values
        # M4 json way to read keys-values
        # S4.6: 
        data_dicts = {}  # Dictionary to store all JSON data, keyed by file path
        keys = []
        for file_path in db_paths:
            # Open the JSON file and parse it into a dictionary
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)  # Properly load the JSON file
                    if isinstance(data, dict):  # Check if the loaded data is a dictionary
                        for key, value in data.items():
                            # Store each key-value pair in the data_dicts dictionary
                            data_dicts[key] = value
                        keys.extend(data.keys())  # Append the list of keys from this JSON
                    else:
                        print(f"Warning: File {file_path} does not contain a JSON object (dictionary)")
                except json.JSONDecodeError:
                    print(f"Error: File {file_path} could not be parsed as JSON.")
        #lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)}')
        
        
        # S4.7: Search for the best performance (reference point) in the database
        res_reference = 0
        max_perf = 0
        
        # Iterate over the JSON data to find the best performance
        for key, value in data_dicts.items():
            #print(value['perf'])
            # Here you can add conditions to check specific keys and their values
            if key.startswith('lv1') or value['perf'] == 0:  # Assuming `value` has a key 'perf'
                continue
            if value['perf'] > max_perf:  # Assuming 'perf' is the key in value dict
                max_perf = value['perf']
                res_reference = value
            if res_reference != 0:
                # saver.log_info(f'reference point for {n} is {res_reference}')
                pass
            else:
                saver.log_info(f'did not find reference point for {n} with {len(keys)} points')
        
        # S4.8-4.14
        # S4.8: Process each key
        cnt = 0
        # Iterate over the JSON data to find the best performance
        for key, v in data_dicts.items():
            
            # S4.9: Skip certain keys based on task type (e.g., regression)
            if FLAGS.task == 'regression' and not FLAGS.invalid and v['perf'] < FLAGS.min_allowed_latency:#TODO: for test dataset, we just need to comment out
                continue
            cnt += 1
            
            # S4.10: Encode graph nodes and edges into dictionaries
            xy_dict = _encode_X_dict(
                g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes = btypes, numerics=numerics, point=v['point'])
            edge_dict = _encode_edge_dict(
                g, ftypes=ftypes_edge, ptypes=ptypes_edge)
        
            # S4.11: Encode pragma features M5 read point
            pragmas = []
            pragma_name = []
            for name, value in sorted(v['point'].items()):
                if type(value) is str:
                    if value.lower() == 'flatten': #### TRY ME: changing the encodering of pipeline pragma to see if it helps with better GAE path
                        value = 100 # 2
                    elif value.lower() == 'off':
                        value = 1
                    elif value.lower() == '':
                        value = 50 # 3
                    else:
                        raise ValueError()
                elif type(value) is int:
                    pass
                else:
                    raise ValueError()
                pragmas.append(value)
                pragma_name.append(name)
            
            
            # S4.12: Ensure that all graphs have the same pragma vector size
            check_dim = init_feat_dict.get(gname)
            if check_dim is not None:
                assert check_dim[0] == len(pragmas), print(check_dim, len(pragmas))
            else:
                init_feat_dict[gname] = [len(pragmas)]
                
            ## same vector size for pragma vector
            pragmas.extend([0] * (max_pragma_length - len(pragmas)))
                
            xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))

            # S4.13: Process regression/classification tasks and prepare targets
            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(v['perf'] + FLAGS.epsilon)
                        elif 'const' in FLAGS.norm_method:
                            y = v['perf'] * FLAGS.normalizer
                            if y == 0:
                                y = FLAGS.max_number * FLAGS.normalizer
                            if FLAGS.norm_method == 'const-log2':
                                y = math.log2(y)
                        elif 'speedup' in FLAGS.norm_method:
                            assert v['perf'] != 0
                            y = FLAGS.normalizer / v['perf']
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y) / 2
                        elif FLAGS.norm_method == 'off':
                            y = v['perf']
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([v['perf']]))
                        xy_dict['kernel_speedup'] = torch.FloatTensor(np.array([math.log2(res_reference['perf'] / v['perf'])]))

                    elif tname == 'quality':
                        y = finte_diff_as_quality(v, res_reference)#M6: replacement ob -> v
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(y + FLAGS.epsilon)
                        elif FLAGS.norm_method == 'const':
                            y = y * FLAGS.normalizer
                        elif FLAGS.norm_method == 'off':
                            pass
                    elif 'util' in tname or 'total' in tname:
                        y = v['res_util'][tname] * FLAGS.util_normalizer
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            elif FLAGS.task == 'class':
                if 'lv1' in key:
                    lv2_key = key.replace('lv1', 'lv2')
                    if lv2_key in keys:
                        continue
                    else:
                        y = 0
                else:
                    y = 0 if v['perf'] < FLAGS.min_allowed_latency else 1    
                xy_dict['perf'] = torch.FloatTensor(np.array([y])).type(torch.LongTensor)
            else:
                raise NotImplementedError()

            # S4.14: Store the variant in the graph
            vname = key

            g.variants[vname] = (xy_dict, edge_dict)
            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']
            
            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']
                

        # S4.15: Log the final number of valid configurations
        saver.log_info(f'final valid: {cnt}')
        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')

    # S5: Train the encoders if they were not pre-loaded
    if FLAGS.encoder_path == None:
        enc_ptype.fit(X_ptype_all)
        enc_ntype.fit(X_ntype_all)
        enc_itype.fit(X_itype_all)
        enc_ftype.fit(X_ftype_all)
        enc_btype.fit(X_btype_all)
        
        enc_ftype_edge.fit(edge_ftype_all)
        enc_ptype_edge.fit(edge_ptype_all)

        saver.log_info(f'Done {num_files} files tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)} {ntypes}')
        saver.log_info(f'\titypes {len(itypes)} {itypes}')
        saver.log_info(f'\tbtypes {len(btypes)} {btypes}')
        saver.log_info(f'\tftypes {len(ftypes)} {ftypes}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')

    # S6: Encode and store final data
    for gname, g in all_gs.items():
        edge_index = create_edge_index(g)
        saver.log_info(f'edge_index created for {gname}')
        new_gname = gname.split('_')[0]
        for vname, d in g.variants.items():
            d_node, d_edge = d
            X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
            edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)

            if FLAGS.task == 'regression':
                data_list.append(Data(
                    gname=new_gname,
                    x=X,
                    key=vname,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    kernel=gname,
                    X_contextnids=d_node['X_contextnids'],
                    X_pragmanids=d_node['X_pragmanids'],                    
                    X_pragmascopenids=d_node['X_pragmascopenids'],                    
                    X_pseudonids=d_node['X_pseudonids'],    
                    X_icmpnids=d_node['X_icmpnids'],    
                    X_pragma_per_node=d_node['X_pragma_per_node'],            
                    pragmas=d_node['pragmas'],
                    perf=d_node['perf'],
                    actual_perf=d_node['actual_perf'],
                    kernel_speedup=d_node['kernel_speedup'], # base is different per kernel
                    quality=d_node['quality'],
                    util_BRAM=d_node['util-BRAM'],
                    util_DSP=d_node['util-DSP'],
                    util_LUT=d_node['util-LUT'],
                    util_FF=d_node['util-FF'],
                    total_BRAM=d_node['total-BRAM'],
                    total_DSP=d_node['total-DSP'],
                    total_LUT=d_node['total-LUT'],
                    total_FF=d_node['total-FF']
                ))
            elif FLAGS.task == 'class':
                data_list.append(Data(
                    gname=new_gname,
                    x=X,
                    key=vname,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    kernel=gname,
                    X_contextnids=d_node['X_contextnids'],
                    X_pragmanids=d_node['X_pragmanids'],
                    X_pragmascopenids=d_node['X_pragmascopenids'],                    
                    X_pseudonids=d_node['X_pseudonids'],    
                    X_icmpnids=d_node['X_icmpnids'],    
                    X_pragma_per_node=d_node['X_pragma_per_node'],
                    pragmas=d_node['pragmas'],
                    perf=d_node['perf']
                ))
            else:
                raise NotImplementedError()


    # S7: Log graph statistics (number of nodes, degrees, etc.)
    nns = [d.x.shape[0] for d in data_list]
    print_stats(nns, 'number of nodes')
    ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
    print_stats(ads, 'avg degrees')
    saver.info(data_list[0])
    saver.log_info(f'dataset[0].num_features {data_list[0].num_features}')
    # S8: Plot distribution of targets and save dataset
    TARGETS.append('actual_perf')
    for target in TARGETS:
        if not hasattr(data_list[0], target.replace('-', '_')):
            saver.warning(f'Data does not have attribute {target}')
            continue
        ys = [_get_y(d, target).item() for d in data_list]
        plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
        saver.log_info(f'{target}_ys', Counter(ys))

    # S9: Save the data to disk if regeneration is forced
    if FLAGS.force_regen:
        saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
        rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        # existing_files = os.listdir(SAVE_DIR)
        # if 'raw' in existing_files:
        #     existing_files.remove('raw')
        # if 'processed' in existing_files:
        #     existing_files.remove('processed')
        # if 'encoders.klepto' in existing_files:
        #     existing_files.remove('encoders.klepto')
        # if 'pragma_dim.klepto' in existing_files:
        #     existing_files.remove('pragma_dim.klepto')
        # existing_indices = [int(ef.split('_')[1][:-3]) for ef in existing_files]
        # existing_max = np.max(existing_indices) + 1
        # print(existing_max)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))
            # torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i + existing_max)))

    # S10: Save the encoder and feature dimension information to disk
    if FLAGS.force_regen:
        from utils import save
        obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype,
            'enc_itype': enc_itype, 'enc_ftype': enc_ftype,
            'enc_btype': enc_btype, 
            'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}
        p = ENCODER_PATH
        save(obj, p)
        
        for gname in init_feat_dict:
            init_feat_dict[gname].append(max_pragma_length)
        name = 'pragma_dim'
        save(init_feat_dict, join(SAVE_DIR, name))
        
        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim[0]}')

    # S11: Return the dataset object and feature dictionary
    rtn = MyOwnDataset()
    return rtn, init_feat_dict


def get_data_list():
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')
    # create a redis database
    # database = redis.StrictRedis(host='localhost', port=4444)
    database = redis.StrictRedis(host='localhost')

    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    if FLAGS.encoder_path != None:
        saver.info(f'loading encoder from {FLAGS.encoder_path}')
        encoders = load(FLAGS.encoder_path, saver.logdir)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']
        
        enc_ftype_edge = encoders['enc_ftype_edge']
        enc_ptype_edge = encoders['enc_ptype_edge']

    else:
        ## handle_unknown='ignore' is crucial for handling unknown variables of new kernels
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')
        
        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')

    all_gs = OrderedDict()
    X_ntype_all = []
    X_ptype_all = []
    X_itype_all = []
    X_ftype_all = []
    X_btype_all = []
    
    edge_ftype_all = []
    edge_ptype_all = []
    tot_configs = 0
    num_files = 0
    init_feat_dict = {}
    max_pragma_length = 0

    if FLAGS.separate_T and FLAGS.pragma_encoder:
        for gexf_file in tqdm(GEXF_FILES[0:]):
            saver.log_info(f'now processing {gexf_file}')
            db_paths = []
            n = basename(gexf_file).split('_')[0]
            for db_p in db_path:
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and FLAGS.v_db in f and f'one-db-extended-round{FLAGS.round_num}' in f] # and not 'updated' in f
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {n}. Skipping.')
                continue
            database.flushdb()
            for idx, file in enumerate(db_paths):
                saver.log_info(f'processing db_paths for {n}: {file}')
                with open(file, 'rb') as f_db:
                    database.hmset(0, pickle.load(f_db))
                break
            keys = [k.decode('utf-8') for k in database.hkeys(0)]
            for key in sorted(keys):
                obj = pickle.loads(database.hget(0, key))
                # try:
                if type(obj) is int or type(obj) is dict:
                    continue
                if FLAGS.task == 'regression' and key[0:3] == 'lv1':# or obj.perf == 0:#obj.ret_code.name == 'PASS':
                    continue
                if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf == 0:
                    continue
                #### TODO !! fix databases that have this problem:
                if obj.point == {}:
                    continue
                len_pragmas = len(obj.point)
                max_pragma_length = max(max_pragma_length, len_pragmas)
                break
    else:
        max_pragma_length = 21

    data_list = []
    for gexf_file in tqdm(GEXF_FILES[0:]):  # TODO: change for partial/full data
        saver.info(f'Working on graph file: {gexf_file}')
        if FLAGS.dataset == 'vitis-cnn':
            if FLAGS.task == 'regression' and FLAGS.tag == 'only-vitis' and 'cnn' in gexf_file:
                continue
            pass
        elif FLAGS.dataset == 'simple-programl':
            pass
        elif FLAGS.dataset == 'machsuite' or 'programl' in FLAGS.dataset:
            proceed = False
            for k in ALL_KERNEL:
                if f'/{k}_processed_result.gexf' in gexf_file:
                    target_kernel = k
                    proceed = True
                    break
            if not proceed:
                saver.info(f'Skipping this file as the kernel name is not selected. Check config file.')
                continue
            # pass
        else:
            raise NotImplementedError()

        g = nx.read_gexf(gexf_file)
        # g = _check_prune_non_pragma_nodes(g)
        g.variants = OrderedDict()
        if FLAGS.dataset == 'simple-programl':
            # gname = basename(dirname(gexf_file))
            # n = basename(dirname(gexf_file))
            gname = basename(gexf_file).split('.')[0]
            n = f"{basename(gexf_file).split('_')[0]}_"
        else:
            gname = basename(gexf_file).split('.')[0]
            n = f"{basename(gexf_file).split('_')[0]}_"
        saver.log_info(gname)
        all_gs[gname] = g

        if FLAGS.dataset == 'vitis-cnn':
            if n != 'cnn1':
                db_paths = glob(f'../dse_database/databases/vitis/exhaustive/{n}_result.db')
                db_paths += glob(f'../dse_database/databases/vitis/bottleneck/{n}_result.db')
            else:
                db_paths = glob(f'../dse_database/databases/cnn_case1/{n}_result*.db')
        elif FLAGS.dataset == 'machsuite':
            db_paths = glob(f'../dse_database/machsuite/databases/exhaustive/{n}_result*.db')
            db_paths += glob(f'../dse_database/machsuite/databases/bottleneck/{n}_result*.db')
        elif FLAGS.dataset == 'simple-programl':
            db_paths = [f for f in iglob(join(get_root_path(), 'dse_database/simple-program/databases/**'), recursive=True) if f.endswith('.db') and n in f and 'one-db' in f]
        elif FLAGS.dataset == 'programl':
            db_paths = []
            for db_p in db_path:
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and FLAGS.v_db in f and f'one-db-extended-round{FLAGS.round_num}' in f] # and 'extended' not in f and 'round' not in f # and 'gae-on' in f] # and not 'updated' in f
                db_paths.extend(paths)
            if db_paths is None:
                saver.warning(f'No database found for {n}. Skipping.')
                continue
        elif FLAGS.dataset == 'programl-machsuite':
            #db_paths_dict = {}
            #for KERNEL in MACHSUITE_KERNEL:
            db_paths = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f]
            #    db_paths_dict[KERNEL] = db_paths
        else:
            raise NotImplementedError()

        # db_paths = sorted(db_paths)
        database.flushdb()
        saver.log_info(f'db_paths for {n}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        assert len(db_paths) == 1
        
        # load the database and get the keys
        # the key for each entry shows the value of each of the pragmas in the source file
        for file in db_paths:
            f_db = open(file, 'rb')
            # print('loading', f_db)
            data = pickle.load(f_db)
            database.hmset(0, data)
            f_db.close()
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')

        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            if pickle_obj is None:
                continue
            obj = pickle.loads(pickle_obj)
            if type(obj) is int or type(obj) is dict:
                continue
            if key[0:3] == 'lv1' or obj.perf == 0:#obj.ret_code.name == 'PASS':
                continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.log_info(f'did not find reference point for {n} with {len(keys)} points')

        cnt = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            if pickle_obj is None:
                continue
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':# or obj.perf == 0:#obj.ret_code.name == 'PASS':
                continue
            if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf < FLAGS.min_allowed_latency:
                continue
            #### TODO !! fix databases that have this problem:
            if obj.point == {}:
                continue
            cnt += 1
            # print(key, obj.point)
            xy_dict = _encode_X_dict(
                g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes=btypes, numerics=numerics, point=obj.point)
            edge_dict = _encode_edge_dict(
                g, ftypes=ftypes_edge, ptypes=ptypes_edge)
            
            
            pragmas = []
            pragma_name = []
            for name, value in sorted(obj.point.items()):
                if type(value) is str:
                    if value.lower() == 'flatten': #### TRY ME: changing the encodering of pipeline pragma to see if it helps with better GAE path
                        value = 100 # 2
                    elif value.lower() == 'off':
                        value = 1
                    elif value.lower() == '':
                        value = 50 # 3
                    else:
                        raise ValueError()
                elif type(value) is int:
                    pass
                else:
                    raise ValueError()
                pragmas.append(value)
                pragma_name.append(name)
            
            # if 'gemver' in gname:
            #     print(len(pragmas), obj.point)
            # if 'gemver' in gname and len(pragmas) == 21:
            #     database.hdel(0, key)
            #     saver.warning(f'deleted {key} from database of {gname}')
            #     assert len(db_paths) == 1
            #     persist(database, db_paths[0])
            #     continue

            check_dim = init_feat_dict.get(gname)
            if check_dim is not None:
                # saver.info((gname, check_dim, len(pragmas)))
                assert check_dim[0] == len(pragmas), print(check_dim, len(pragmas))
                # if check_dim == len(pragmas):
                #     pass
                # else:
                #     database.hdel(0, key)
                #     print(check_dim, len(pragmas))
                #     saver.warning(f'deleted {key} from database of {gname}')
                #     assert len(db_paths) == 1
                #     persist(database, db_paths[0])
                #     continue
            else:
                init_feat_dict[gname] = [len(pragmas)]
            if FLAGS.pragma_uniform_encoder:
                pragmas.extend([0] * (max_pragma_length - len(pragmas)))
                
            xy_dict['pragmas'] = torch.FloatTensor(np.array([pragmas]))


            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(obj.perf + FLAGS.epsilon)
                        elif 'const' in FLAGS.norm_method:
                            y = obj.perf * FLAGS.normalizer
                            if y == 0:
                                y = FLAGS.max_number * FLAGS.normalizer
                            if FLAGS.norm_method == 'const-log2':
                                y = math.log2(y)
                        elif 'speedup' in FLAGS.norm_method:
                            assert obj.perf != 0
                            y = FLAGS.normalizer / obj.perf
                            # y = res_reference.perf / obj.perf
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y) / 2
                        elif FLAGS.norm_method == 'off':
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                        xy_dict['kernel_speedup'] = torch.FloatTensor(np.array([math.log2(res_reference.perf / obj.perf)]))

                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(y + FLAGS.epsilon)
                        elif FLAGS.norm_method == 'const':
                            y = y * FLAGS.normalizer
                        elif FLAGS.norm_method == 'off':
                            pass
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname] * FLAGS.util_normalizer
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            elif FLAGS.task == 'class':
                if 'lv1' in key:
                    lv2_key = key.replace('lv1', 'lv2')
                    if lv2_key in keys:
                        continue
                    else:
                        y = 0
                else:
                    y = 0 if obj.perf < FLAGS.min_allowed_latency else 1    
                xy_dict['perf'] = torch.FloatTensor(np.array([y])).type(torch.LongTensor)
            else:
                raise NotImplementedError()


            vname = key

            g.variants[vname] = (xy_dict, edge_dict)
            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']
            
            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']
                

        saver.log_info(f'final valid: {cnt}')
        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')
        # saver.log_info(f"\tntypes {len(xy_dict['X_ntype'])} {xy_dict['X_ntype']}")
        saver.log_info(f'\tntypes {len(ntypes)} {ntypes}')
        saver.log_info(f'\titypes {len(itypes)} {itypes}')
        saver.log_info(f'\tbtypes {len(btypes)} {btypes}')
        saver.log_info(f'\tftypes {len(ftypes)} {ftypes}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')

    if FLAGS.encoder_path == None:
        enc_ptype.fit(X_ptype_all)
        enc_ntype.fit(X_ntype_all)
        enc_itype.fit(X_itype_all)
        enc_ftype.fit(X_ftype_all)
        enc_btype.fit(X_btype_all)
        
        enc_ftype_edge.fit(edge_ftype_all)
        enc_ptype_edge.fit(edge_ptype_all)

        saver.log_info(f'Done {num_files} files tot_configs {tot_configs}')
        # saver.log_info(f'\tntypes {len(X_ntype_all)} {X_ntype_all}')
        saver.log_info(f'\tntypes {len(ntypes)} {ntypes}')
        saver.log_info(f'\titypes {len(itypes)} {itypes}')
        saver.log_info(f'\tbtypes {len(btypes)} {btypes}')
        saver.log_info(f'\tftypes {len(ftypes)} {ftypes}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')

    kernel_indices = {} # key: kernel name, value: (start_pos, size)
    count = 0
    for gname, g in all_gs.items():
        edge_index = create_edge_index(g)
        saver.log_info(f'edge_index created for {gname}')
        new_gname = gname.split('_')[0]
        if new_gname not in kernel_indices:
            kernel_indices[new_gname] = (count, len(g.variants.items()))
            count += len(g.variants.items())
        for vname, d in g.variants.items():
            d_node, d_edge = d
            X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
            edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)

            if FLAGS.task == 'regression':
                data_list.append(Data(
                    gname=new_gname,
                    key=vname,
                    X_contextnids=d_node['X_contextnids'],
                    X_pragmanids=d_node['X_pragmanids'],                    
                    X_pragmascopenids=d_node['X_pragmascopenids'],                    
                    X_pseudonids=d_node['X_pseudonids'],    
                    X_icmpnids=d_node['X_icmpnids'],    
                    X_pragma_per_node=d_node['X_pragma_per_node'],            
                    x=X,
                    edge_index=edge_index,
                    pragmas=d_node['pragmas'],
                    perf=d_node['perf'],
                    actual_perf=d_node['actual_perf'],
                    kernel_speedup=d_node['kernel_speedup'], # base is different per kernel
                    quality=d_node['quality'],
                    util_BRAM=d_node['util-BRAM'],
                    util_DSP=d_node['util-DSP'],
                    util_LUT=d_node['util-LUT'],
                    util_FF=d_node['util-FF'],
                    total_BRAM=d_node['total-BRAM'],
                    total_DSP=d_node['total-DSP'],
                    total_LUT=d_node['total-LUT'],
                    total_FF=d_node['total-FF'],
                    edge_attr=edge_attr,
                    kernel=gname
                ))
            elif FLAGS.task == 'class':
                data_list.append(Data(
                    gname=new_gname,
                    x=X,
                    key=vname,
                    edge_index=edge_index,
                    pragmas=d_node['pragmas'],
                    perf=d_node['perf'],
                    edge_attr=edge_attr,
                    kernel=gname,
                    X_contextnids=d_node['X_contextnids'],
                    X_pragmanids=d_node['X_pragmanids'],
                    X_pragmascopenids=d_node['X_pragmascopenids'],                    
                    X_pseudonids=d_node['X_pseudonids'],    
                    X_icmpnids=d_node['X_icmpnids'],    
                    X_pragma_per_node=d_node['X_pragma_per_node']
                ))
            else:
                raise NotImplementedError()

    nns = [d.x.shape[0] for d in data_list]
    print_stats(nns, 'number of nodes')
    ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
    print_stats(ads, 'avg degrees')
    saver.info(data_list[0])
    saver.log_info(f'dataset[0].num_features {data_list[0].num_features}')
    TARGETS.append('actual_perf')
    for target in TARGETS:
        if not hasattr(data_list[0], target.replace('-', '_')):
            saver.warning(f'Data does not have attribute {target}')
            continue
        ys = [_get_y(d, target).item() for d in data_list]
        # if target == 'quality':
        #     continue
        plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
        saver.log_info(f'{target}_ys', Counter(ys))

    if FLAGS.force_regen:
        saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
        rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        # existing_files = os.listdir(SAVE_DIR)
        # if 'raw' in existing_files:
        #     existing_files.remove('raw')
        # if 'processed' in existing_files:
        #     existing_files.remove('processed')
        # if 'encoders.klepto' in existing_files:
        #     existing_files.remove('encoders.klepto')
        # if 'pragma_dim.klepto' in existing_files:
        #     existing_files.remove('pragma_dim.klepto')
        # existing_indices = [int(ef.split('_')[1][:-3]) for ef in existing_files]
        # existing_max = np.max(existing_indices) + 1
        # print(existing_max)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))
            # torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i + existing_max)))

    if FLAGS.force_regen:
        from utils import save
        # if FLAGS.only_pragma:
        #     obj = {'enc_ptype': enc_ptype}
        # else:
        obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype,
            'enc_itype': enc_itype, 'enc_ftype': enc_ftype,
            'enc_btype': enc_btype, 
            'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}
        p = ENCODER_PATH
        # if FLAGS.encoder_path == None:
        save(obj, p)
        
        if FLAGS.pragma_uniform_encoder:
            for gname in init_feat_dict:
                init_feat_dict[gname].append(max_pragma_length)
        name = 'pragma_dim'
        save(init_feat_dict, join(SAVE_DIR, name))
        
        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim[0]}')

    rtn = MyOwnDataset()
    return rtn, init_feat_dict


def _get_y(data, target):
    return getattr(data, target.replace('-', '_'))

def print_data_stats(data_loader, tvt):
    nns, ads, ys = [], [], []
    for d in tqdm(data_loader):
        nns.append(d.x.shape[0])
        # ads.append(d.edge_index.shape[1] / d.x.shape[0])
        ys.append(d.y.item())
    print_stats(nns, f'{tvt} number of nodes')
    # print_stats(ads, f'{tvt} avg degrees')
    plot_dist(ys, f'{tvt} ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
    saver.log_info(f'{tvt} ys', Counter(ys))


def load_all_gs(remove_all_pragma_nodes):
    rtn = []
    for gexf_file in tqdm(GEXF_FILES[0:]):  # TODO: change for partial/full data
        g = nx.read_gexf(gexf_file)
        rtn.append(g)
        if remove_all_pragma_nodes:
            before = g.number_of_nodes()
            nodes_to_remove = []
            for node, ndata in g.nodes(data=True):
                if 'pragma' in ndata['full_text']:
                    nodes_to_remove.append(node)
            g.remove_nodes_from(nodes_to_remove)
            print(f'Removed {len(nodes_to_remove)} pragma nodes;'
                  f' before {before} now {g.number_of_nodes}')
    return rtn


def load_encoders():
    from utils import load
    rtn = load(ENCODER_PATH, saver.logdir)
    return rtn

def find_pragma_node(g, nid):
    pragma_nodes = {}
    for neighbor in g.neighbors(str(nid)):
        for pragma in ['pipeline', 'parallel', 'tile']:
            if g.nodes[neighbor]['text'].lower() == pragma:
                pragma_nodes[pragma] = neighbor
                break
    
    return pragma_nodes


def get_pragma_numeric(pragma_text, point, pragma_type):
    t_li = pragma_text.split(' ')
    reduction = 0
    for i in range(len(t_li)):
        if 'REDUCTION' in t_li[i].upper(): 
            reduction = 1
        elif 'AUTO{' in t_li[i].upper():
            # print(t_li[i])
            auto_what = _in_between(t_li[i], '{', '}')
            numeric = point[auto_what]
            if type(numeric) is not int: ## '', 'off', 'flatten'
                assert pragma_type.lower() == 'pipeline'
                if numeric == 'flatten':
                    numeric = 10
                elif numeric == 'off':
                    numeric = 1
                else:
                    numeric = 5
            
    return reduction, numeric

def fill_pragma_vector(g, neighbor_pragmas, pragma_vector, point, node):
    '''
        # for each node, a vector of [tile factor, pipeline type, parallel type, parallel factor] 
        # pipeline type: 1: off, 5: cg, 10: flatten
        # parallel type: 1: normal, 2: reduction
        # if no pragma assigned to node, a vector of [0, 0, 0, 0]
    '''
    vector_id = {'pipeline': 1, 'parallel': 3, 'tile': 0}
    for pragma in ['pipeline', 'parallel', 'tile']:
        if pragma in neighbor_pragmas:
            nid = neighbor_pragmas[pragma]
            pragma_text = g.nodes[nid]['full_text']
            reduction, numeric = get_pragma_numeric(pragma_text, point, pragma_type=pragma)
            pragma_vector[vector_id[pragma]] = numeric
            if pragma == 'parallel':
                if reduction == 0:
                    pragma_vector[vector_id[pragma] - 1] = 1
                else:
                    pragma_vector[vector_id[pragma] - 1] = 2
    # saver.log_info(f'point: {point}')
    # saver.log_info(f'{node}, {pragma_vector}')
    return pragma_vector


def encode_g_torch(g, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    x_dict = _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, eftypes=None, btypes=None, point=None)

    X = _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)

    edge_index = create_edge_index(g)

    return X, edge_index


# def _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, ftypes=None, btypes=None, point=None):
#     X_ntype = [] # node type <attribute id="3" title="type" type="long" />
#     X_ptype = [] # pragma type
#     X_numeric = []
#     X_itype = [] # instruction type (text) <attribute id="2" title="text" type="string" />
#     X_ftype = [] # function type <attribute id="1" title="function" type="long" />
#     X_btype = [] # block type <attribute id="0" title="block" type="long" />
#     X_contextnids = [] # 0 or 1 showing context node
#     X_pragmanids = [] # 0 or 1 showing pragma node
#     X_pseudonids = [] # 0 or 1 showing pseudo node
#     X_icmpnids = [] # 0 or 1 showing icmp node
#     ## for pragma as MLP
#     X_pragma_per_node = [] # for each node, a vector of [tile factor, pipeline type, parallel type, parallel factor] 
#                            # pipeline type: 1: off, 5: cg, 10: flatten
#                            # parallel type: 1: normal, 2: reduction
#                            # if no pragma assigned to node, a vector of [0, 0, 0, 0]
#     X_pragmascopenids = [] # 0 or 1 showing if previous vector is all zero or not
    
    
      
#     for nid, (node, ndata) in enumerate(g.nodes(data=True)):  # TODO: node ordering
#         # print(node['type'], type(node['type']))
#         assert nid == int(node), f'{nid} {node}'
#         # print(node['type'], type(node['type']))
#         if ntypes is not None:
#             ntypes[ndata['type']] += 1
#         if itypes is not None:
#             itypes[ndata['text']] += 1
#         if btypes is not None:
#             btypes[ndata['block']] += 1
#         if ftypes is not None:
#             ftypes[ndata['function']] += 1
            
#         pragma_vector = [0, 0, 0, 0]
#         if 'pseudo' in ndata['text']:
#             X_pseudonids.append(1)
#             ## for pragma as MLP
#             if FLAGS.pragma_scope == 'block':
#                 ## check if the block incules any pragma nodes
#                 neighbor_pragmas = find_pragma_node(g, node)
#                 if len(neighbor_pragmas) == 0:
#                     X_pragmascopenids.append(0)
#                 else:
#                     X_pragmascopenids.append(1)
#                     pragma_vector = fill_pragma_vector(g, neighbor_pragmas, pragma_vector, point, node)
#             else: ## other pragma scopes are not implemented yet
#                 raise NotImplementedError()
#         else:
#             X_pseudonids.append(0)
#             X_pragmascopenids.append(0)
#         ## for pragma as MLP: a vector of [tile factor, pipeline type, parallel type, parallel factor]
#         X_pragma_per_node.append(pragma_vector)

#         numeric = 0

#         if 'full_text' in ndata and 'icmp' in ndata['full_text']:
#             cmp_t = ndata['full_text'].split(',')[-1]
#             cmp_t = cmp_t.strip()
#             if cmp_t.isdigit():
#                 cmp_t = eval(cmp_t)
#                 # saver.log_info(cmp_t)
#                 numeric = cmp_t
#                 X_icmpnids.append(1)
#             else:
#                 X_icmpnids.append(0)
#                 pass
#         else:
#             X_icmpnids.append(0)

#         if 'full_text' in ndata and 'pragma' in ndata['full_text']:
#             # print(ndata['content'])
#             p_text = ndata['full_text'].rstrip()
#             assert p_text[0:8] == '#pragma '
#             p_text_type = p_text[8:].upper()

#             if _check_any_in_str(NON_OPT_PRAGMAS, p_text_type):
#                 p_text_type = 'None'
#             else:
#                 if _check_any_in_str(WITH_VAR_PRAGMAS, p_text_type):
#                     # HLS DEPENDENCE VARIABLE=CSIYIY ARRAY INTER FALSE
#                     # HLS DEPENDENCE VARIABLE=<> ARRAY INTER FALSE
#                     t_li = p_text_type.split(' ')
#                     for i in range(len(t_li)):
#                         if 'VARIABLE=' in t_li[i]:
#                             t_li[i] = 'VARIABLE=<>'
#                         elif 'DEPTH=' in t_li[i]:
#                             t_li[i] = 'DEPTH=<>'  # TODO: later add back
#                         elif 'DIM=' in t_li[i]:
#                             numeric = int(t_li[i][4:])
#                             t_li[i] = 'DIM=<>'
#                         elif 'LATENCY=' in t_li[i]:
#                             numeric = int(t_li[i][8:])
#                             t_li[i] = 'LATENCY=<>'
#                     p_text_type = ' '.join(t_li)

#                 pragma_shortened = []
#                 if point is not None:
#                     t_li = p_text_type.split(' ')
#                     skip_next_two = 0
#                     for i in range(len(t_li)):
#                         if skip_next_two == 2:
#                             if t_li[i] == '=':
#                                 skip_next_two = 1
#                                 continue
#                             else:
#                                 skip_next_two = 0
#                         elif skip_next_two == 1:
#                             skip_next_two = 0
#                             continue
#                         if 'REDUCTION' in t_li[i]: ### NEW: use one type for all reductions (previously reduction=D and reduction=C were different)
#                             # saver.info(t_li[i])
#                             if FLAGS.keep_pragma_attribute: ## see reduction as a different kind of parallelization
#                                 pragma_shortened.append('REDUCTION')
#                             skip_next_two = 2
#                         # elif 'PARALLEL' in t_li[i]:
#                         #     pragma_shortened.append('PRALLEL REDUCTION')
#                         elif not FLAGS.keep_pragma_attribute and 'PIPELINE' in t_li[i]: ## see all the pipeline option as the same
#                             pragma_shortened.append(t_li[i])
#                             break
#                         elif 'AUTO{' in t_li[i]:
#                             # print(t_li[i])
#                             auto_what = _in_between(t_li[i], '{', '}')
#                             numeric = point[auto_what]
#                             if type(numeric) is not int:
#                                 t_li[i] = numeric
#                                 pragma_shortened.append(numeric)
#                                 numeric = 0  # TODO: ? '', 'off', 'flatten'
#                             else:
#                                 t_li[i] = 'AUTO{<>}'
#                                 pragma_shortened.append('AUTO{<>}')
#                             break
#                         else:
#                             pragma_shortened.append(t_li[i])
#                     # p_text_type = ' '.join(t_li)
#                     # if len(t_li) != len(pragma_shortened): saver.log_info(f'{t_li} vs {pragma_shortened}')
#                     p_text_type = ' '.join(pragma_shortened)
#                 else:
#                     assert 'AUTO' not in p_text_type
#                 # t = ' '.join(t.split(' ')[0:2])
#             # print('@@@@@', t)
#             if not FLAGS.keep_pragma_attribute: ## see all the pragma options as the same
#                 numeric = 1
#             ptype = p_text_type
#             X_pragmanids.append(1)
#             X_contextnids.append(0)
#         else:
#             ptype = 'None'
#             X_pragmanids.append(0)
#             ## exclude pseudo nodes from context nodes
#             if 'pseudo' in ndata['text']:
#                 X_contextnids.append(0)
#             else:
#                 X_contextnids.append(1)
                
#         if ptypes is not None:
#             ptypes[ptype] += 1
#         if numerics is not None:
#             numerics[numeric] += 1

#         X_ntype.append([ndata['type']])
#         X_ptype.append([ptype])
#         X_numeric.append([numeric])
#         X_itype.append([ndata['text']])
#         X_ftype.append([ndata['function']])
#         X_btype.append([ndata['block']])
        
#     # vname = key

#     X_pragma_per_node = transform_X_torch(X_pragma_per_node)
#     return {'X_ntype': X_ntype, 'X_ptype': X_ptype,
#             'X_numeric': X_numeric, 'X_itype': X_itype,
#             'X_ftype': X_ftype, 'X_btype': X_btype,
#             'X_contextnids': torch.FloatTensor(np.array(X_contextnids)),
#             'X_pragmanids': torch.FloatTensor(np.array(X_pragmanids)),
#             'X_pragmascopenids': torch.FloatTensor(np.array(X_pragmascopenids)),
#             'X_pseudonids': torch.FloatTensor(np.array(X_pseudonids)),
#             'X_icmpnids': torch.FloatTensor(np.array(X_icmpnids)),
#             'X_pragma_per_node': X_pragma_per_node
#             }


def _encode_X_dict(g, ntypes=None, ptypes=None, numerics=None, itypes=None, ftypes=None, btypes=None, point=None, device=FLAGS.device, force_keep_pragma_attribute=False):
    X_ntype = [] # node type <attribute id="3" title="type" type="long" />
    X_ptype = [] # pragma type
    X_numeric = []
    X_itype = [] # instruction type (text) <attribute id="2" title="text" type="string" />
    X_ftype = [] # function type <attribute id="1" title="function" type="long" />
    X_btype = [] # block type <attribute id="0" title="block" type="long" />
    X_contextnids = [] # 0 or 1 showing context node
    X_pragmanids = [] # 0 or 1 showing pragma node
    X_pseudonids = [] # 0 or 1 showing pseudo node
    X_icmpnids = [] # 0 or 1 showing icmp node
    ## for pragma as MLP
    X_pragma_per_node = [] # for each node, a vector of [tile factor, pipeline type, parallel type, parallel factor] 
                           # pipeline type: 1: off, 5: cg, 10: flatten
                           # parallel type: 1: normal, 2: reduction
                           # if no pragma assigned to node, a vector of [0, 0, 0, 0]
    X_pragmascopenids = [] # 0 or 1 showing if previous vector is all zero or not
    
    for nid, (node, ndata) in enumerate(g.nodes(data=True)):  # TODO: node ordering
        # print(node['type'], type(node['type']))
        assert nid == int(node), f'{nid} {node}'
        # print(node['type'], type(node['type']))
        if ntypes is not None:
            ntypes[ndata['type']] += 1
        if itypes is not None:
            itypes[ndata['text']] += 1
        if btypes is not None:
            btypes[ndata['block']] += 1
        if ftypes is not None:
            ftypes[ndata['function']] += 1
            
        pragma_vector = [0, 0, 0, 0]
        if 'pseudo' in ndata['text']:
            X_pseudonids.append(1)
            ## for pragma as MLP
            if FLAGS.pragma_scope == 'block':
                ## check if the block incules any pragma nodes
                neighbor_pragmas = find_pragma_node(g, node)
                if len(neighbor_pragmas) == 0:
                    X_pragmascopenids.append(0)
                else:
                    X_pragmascopenids.append(1)
                    pragma_vector = fill_pragma_vector(g, neighbor_pragmas, pragma_vector, point, node)
            else: ## other pragma scopes are not implemented yet
                raise NotImplementedError()
        else:
            X_pseudonids.append(0)
            X_pragmascopenids.append(0)
        ## for pragma as MLP: a vector of [tile factor, pipeline type, parallel type, parallel factor]
        X_pragma_per_node.append(pragma_vector)

        numeric = 0

        if 'full_text' in ndata and 'icmp' in ndata['full_text']:
            cmp_t = ndata['full_text'].split(',')[-1]
            cmp_t = cmp_t.strip()
            if cmp_t.isdigit():
                cmp_t = eval(cmp_t)
                # saver.log_info(cmp_t)
                numeric = cmp_t
                X_icmpnids.append(1)
            else:
                X_icmpnids.append(0)
                pass
        else:
            X_icmpnids.append(0)

        if 'full_text' in ndata and 'pragma' in ndata['full_text']:
            # print(ndata['content'])
            p_text = ndata['full_text'].rstrip()
            assert p_text[0:8] == '#pragma '
            p_text_type = p_text[8:].upper()

            if _check_any_in_str(NON_OPT_PRAGMAS, p_text_type):
                p_text_type = 'None'
            else:
                if _check_any_in_str(WITH_VAR_PRAGMAS, p_text_type):
                    # HLS DEPENDENCE VARIABLE=CSIYIY ARRAY INTER FALSE
                    # HLS DEPENDENCE VARIABLE=<> ARRAY INTER FALSE
                    t_li = p_text_type.split(' ')
                    for i in range(len(t_li)):
                        if 'VARIABLE=' in t_li[i]:
                            t_li[i] = 'VARIABLE=<>'
                        elif 'DEPTH=' in t_li[i]:
                            t_li[i] = 'DEPTH=<>'  # TODO: later add back
                        elif 'DIM=' in t_li[i]:
                            numeric = int(t_li[i][4:])
                            t_li[i] = 'DIM=<>'
                        elif 'LATENCY=' in t_li[i]:
                            numeric = int(t_li[i][8:])
                            t_li[i] = 'LATENCY=<>'
                    p_text_type = ' '.join(t_li)

                pragma_shortened = []
                if point is not None:
                    t_li = p_text_type.split(' ')
                    skip_next_two = 0
                    for i in range(len(t_li)):
                        if skip_next_two == 2:
                            if t_li[i] == '=':
                                skip_next_two = 1
                                continue
                            else:
                                skip_next_two = 0
                        elif skip_next_two == 1:
                            skip_next_two = 0
                            continue
                        if 'REDUCTION' in t_li[i]: ### NEW: use one type for all reductions (previously reduction=D and reduction=C were different)
                            # saver.info(t_li[i])
                            if FLAGS.keep_pragma_attribute or force_keep_pragma_attribute: ## see reduction as a different kind of parallelization
                                pragma_shortened.append('REDUCTION')
                            skip_next_two = 2
                        # elif 'PARALLEL' in t_li[i]:
                        #     pragma_shortened.append('PRALLEL REDUCTION')
                        elif not (FLAGS.keep_pragma_attribute or force_keep_pragma_attribute) and 'PIPELINE' in t_li[i]: ## see all the pipeline option as the same
                            pragma_shortened.append(t_li[i])
                            break
                        elif 'AUTO{' in t_li[i]:
                            # print(t_li[i])
                            auto_what = _in_between(t_li[i], '{', '}')
                            numeric = point[auto_what]
                            if type(numeric) is not int:
                                t_li[i] = numeric
                                pragma_shortened.append(numeric)
                                numeric = 0  # TODO: ? '', 'off', 'flatten'
                            else:
                                t_li[i] = 'AUTO{<>}'
                                pragma_shortened.append('AUTO{<>}')
                            break
                        else:
                            pragma_shortened.append(t_li[i])
                    # p_text_type = ' '.join(t_li)
                    # if len(t_li) != len(pragma_shortened): saver.log_info(f'{t_li} vs {pragma_shortened}')
                    p_text_type = ' '.join(pragma_shortened)
                else:
                    assert 'AUTO' not in p_text_type
                # t = ' '.join(t.split(' ')[0:2])
            # print('@@@@@', t)
            if not (FLAGS.keep_pragma_attribute or force_keep_pragma_attribute): ## see all the pragma options as the same
                numeric = 1
            ptype = p_text_type
            X_pragmanids.append(1)
            X_contextnids.append(0)
        else:
            ptype = 'None'
            X_pragmanids.append(0)
            ## exclude pseudo nodes from context nodes
            if 'pseudo' in ndata['text']:
                X_contextnids.append(0)
            else:
                X_contextnids.append(1)
                
        if ptypes is not None:
            ptypes[ptype] += 1
        if numerics is not None:
            numerics[numeric] += 1

        X_ntype.append([ndata['type']])
        X_ptype.append([ptype])
        X_numeric.append([numeric])
        X_itype.append([ndata['text']])
        X_ftype.append([ndata['function']])
        X_btype.append([ndata['block']])
        
    # vname = key

    X_pragma_per_node = transform_X_torch(X_pragma_per_node, device)
    return {'X_ntype': X_ntype, 'X_ptype': X_ptype,
            'X_numeric': X_numeric, 'X_itype': X_itype,
            'X_ftype': X_ftype, 'X_btype': X_btype,
            'X_contextnids': torch.FloatTensor(np.array(X_contextnids)),
            'X_pragmanids': torch.FloatTensor(np.array(X_pragmanids)),
            'X_pragmascopenids': torch.FloatTensor(np.array(X_pragmascopenids)),
            'X_pseudonids': torch.FloatTensor(np.array(X_pseudonids)),
            'X_icmpnids': torch.FloatTensor(np.array(X_icmpnids)),
            'X_pragma_per_node': X_pragma_per_node
            }


def transform_X_torch(X, device=FLAGS.device):
    X = torch.FloatTensor(np.array(X))
    X = coo_matrix(X)
    X = _coo_to_sparse(X, device)
    X = X.to_dense()
    return X

def _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype, device=FLAGS.device):
    """
    x_dict is the returned dict by _encode_X_dict()
    """
    X_ntype = enc_ntype.transform(x_dict['X_ntype'])
    X_ptype = enc_ptype.transform(x_dict['X_ptype'])
    X_itype = enc_itype.transform(x_dict['X_itype'])
    X_ftype = enc_ftype.transform(x_dict['X_ftype'])
    X_btype = enc_btype.transform(x_dict['X_btype'])

    X_numeric = x_dict['X_numeric']
    # print(len(enc_ntype.categories_[0]))
    # print(len(X_numeric))
    # saver.log_info(X_ntype.shape(0), X_ptype.shape(0), X_itype.shape(0), X_ftype.shape(0), X_btype.shape(0)) #X_numeric.shape(0))
    if FLAGS.no_pragma:
        X = X_ntype
        X = X.toarray()
        X = torch.FloatTensor(X)
    else:
        X = hstack((X_ntype, X_ptype, X_numeric, X_itype, X_ftype, X_btype))
        X = _coo_to_sparse(X, device)
        X = X.to_dense()
    return X





def _encode_edge_dict(g, ftypes=None, ptypes=None):
    X_ftype = [] # flow type <attribute id="5" title="flow" type="long" />
    X_ptype = [] # position type <attribute id="6" title="position" type="long" />    
      
    for nid1, nid2, edata in g.edges(data=True):  # TODO: node ordering
        X_ftype.append([edata['flow']])
        X_ptype.append([edata['position']])

    return {'X_ftype': X_ftype, 'X_ptype': X_ptype}

    
def _encode_edge_torch(edge_dict, enc_ftype, enc_ptype, device=FLAGS.device, rm_index=False):
    """
    edge_dict is the dictionary returned by _encode_edge_dict
    """
    X_ftype = enc_ftype.transform(edge_dict['X_ftype'])
    X_ptype = enc_ptype.transform(edge_dict['X_ptype'])

    if FLAGS.encode_edge_position:
        X = hstack((X_ftype, X_ptype), format='coo')
    else:
        X = coo_matrix(X_ftype)
    X = _coo_to_sparse(X, device)
    X = X.to_dense()

    if X.shape[0] == 9828:
        X = torch.concat([X[0:8434], X[8435:8454], X[8455:]])
    return X
        

def _in_between(text, left, right):
    # text = 'I want to find a string between two substrings'
    # left = 'find a '
    # right = 'between two'
    return text[text.index(left) + len(left):text.index(right)]


def _check_any_in_str(li, s):
    for li_item in li:
        if li_item in s:
            return True
    return False


def create_edge_index(g, device=FLAGS.device):
    g = nx.convert_node_labels_to_integers(g)
    # es = list(g.edges)
    # if len(es[0]) == 3:
    #     vis_edge = []
    #     for e in es:
    #         if e[0:2] in es:
    #             g.remove_edge(e[0], e[1])
    #         else:
    #             vis_edge.append(e[0:2])
    edge_index = torch.LongTensor(list(g.edges)).t().contiguous().to(device)
    if edge_index.shape[0] == 3:
        edge_index = edge_index[0:2]
        edge_index = torch.concat([edge_index[:, 0:8434], edge_index[:, 8435:8454], edge_index[:, 8455:]], 1)
        assert edge_index.shape[1] == 9826
        assert len(g) == 2685
    return edge_index


def _coo_to_sparse(coo, device=FLAGS.device):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = coo.shape

    rtn = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return rtn


def _check_prune_non_pragma_nodes(g):
    if FLAGS.only_pragma:
        to_remove = []
        for node, ndata in g.nodes(data=True):
            x = ndata.get('full_text')
            if x is None:
                x = ndata['type']
            if type(x) is not str or (not 'Pragma' in x and not 'pragma' in x):
                to_remove.append(node)
        before = g.number_of_nodes()
        g.remove_nodes_from(to_remove)
        saver.log_info(f'Removed {len(to_remove)} non-pragma nodes from G -'
                       f'- {before} to {g.number_of_nodes()}')
        assert g.number_of_nodes() + len(to_remove) == before
    return g
