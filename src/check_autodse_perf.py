from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch
import os
from glob import glob, iglob
from os.path import join
import os.path
from config import FLAGS
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from parameter import gen_key_from_design_point, get_default_point, compile_design_space
from dse import ExhaustiveExplorer
from RL import dse_utils
from saver import saver
from utils import get_save_path


check_kernels = ['3d-rendering', 'att-3mm', 'att-3mm-fuse', 'knn', 'spam-filter', 'vmmv']


#assert FLAGS.dse_kernel in check_kernels
SAVE_DIR = '../save/programl/v21_MLP-True-round1-40kernel-icmp-feb-db-extended-pseudo-block-connected-hierarchy-regression_edge-position-True_norm_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF'
SAVE_DIR = SAVE_DIR + '/' + FLAGS.dse_kernel
dse_kernel = FLAGS.dse_kernel


class SimpleDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None, e_kernel=None):
        # self.processed_dir = PROCESSED_DIR
        super(SimpleDataset, self).__init__(SAVE_DIR, transform, pre_transform)
        if data_files is not None:
            if FLAGS.load_data_to_device:
                data_files = [torch.load(df).to(FLAGS.device) for df in data_files]
            self.data_files = data_files
        self.file_list = self.processed_file_names
        # for i in range(0, len(file_list), 10):
        #     data = torch.load(file_list[i])
        # self.SAVE_DIR = SAVE_DIR
        self.fail_cnt = 0

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
        #     torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
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
            fn = os.path.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        return fn

    def get(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = self.file_list[idx]
        if FLAGS.load_data_to_device == False or hasattr(self, 'data_files') == False:
            try:
                data = torch.load(fn)
            except:
                fn = os.path.join(SAVE_DIR, 'data_{}.pt'.format(2353))  # For the v21_granular finetune data
                data = torch.load(fn)
                if self.fail_cnt == 3:
                    print('fail 3 times extracting data! (data.py)')
                    print(fn)
                    exit()
                self.fail_cnt += 1
        else:
            data = self.data_files[idx]
        return data


def calculate_overlap_perf(plot = False, search_limit = 75000):
    dataset = SimpleDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    key_list = []
    perf_list = []
    overlap_perf_list = []

    for _, data in enumerate(tqdm(loader)):
        config_str = data[0].key
        if dse_kernel not in check_kernels:    # some kernels have 'lv2:' at the begining of the string
            assert config_str[0:4] in ['lv1:', 'lv2:']
            config_str = config_str[4:]
        else:
            assert config_str[0:4] not in ['lv1:', 'lv2:']
        lut_mask = (data[0]['util_LUT'] <= 0.8)
        ff_mask = (data[0]['util_FF'] <= 0.8)
        dsp_mask = (data[0]['util_DSP'] <= 0.8)
        bram_mask = (data[0]['util_BRAM'] <= 0.8)
        mask = lut_mask * ff_mask * dsp_mask * bram_mask
        if mask:
            key_list.append(config_str)
            perf_list.append(data[0].actual_perf)
    mean_perf = np.mean(perf_list)
    best_perf = np.min(perf_list)
    print("--------------- FINISHED PROCESSING AUTODSE DATA ----------------")

    explorer = ExhaustiveExplorer('../dse_database/poly/config/', dse_kernel, '../dse_database/programl/poly/processed', {})
    explored_len = 0
    tqdm_bar = tqdm(search_limit)
    cnt = 0
    overlap_perf_prefix = []
    overlap_idx_list = []

    for batch in explorer.gen():
        if explored_len > search_limit:
            break
        for batch_i, point in enumerate(batch):
            key = gen_key_from_design_point(point)
            for i, autodse_key in enumerate(key_list):
                if (autodse_key == key):
                    cnt += 1
                    overlap_perf_list.append(perf_list[i])
                    overlap_perf_prefix.append(np.min(overlap_perf_list))
                    overlap_idx_list.append(explored_len + batch_i)
        tqdm_bar.update(len(batch))
        explored_len += len(batch)
    mean_overlap_perf = np.mean(overlap_perf_list)
    best_overlap_perf = np.min(overlap_perf_list)

    print(f'\n#Overlapping: {cnt} (total {len(perf_list)})')
    print(f'Autodse mean perf: {mean_perf}')
    print(f'Autodse best perf: {best_perf}')
    print(f'Overlap mean perf: {mean_overlap_perf}')
    print(f'Overlap best perf: {best_overlap_perf}')
    if plot == True:
        plot_overlap_perf(overlap_idx_list, overlap_perf_prefix, best_perf, search_limit)


# Plot the relation between exhaustive search range and best overlapping performance
def plot_overlap_perf(overlap_idx_list, overlap_perf_prefix, best_perf, search_limit):
    overlap_idx_list.append(search_limit)
    overlap_perf_prefix.append(overlap_perf_prefix[-1])
    overlap_perf_prefix = best_perf / np.array(overlap_perf_prefix)
    plt.figure(figsize=(8, 6))
    plt.title(dse_kernel, size=20)
    plt.plot(overlap_idx_list, overlap_perf_prefix)
    plt.axvline(75000, linestyle=':', color='black')
    plt.savefig(f'overlap_{dse_kernel}.png', bbox_inches='tight')


# Observe the hueristics DFS order of pragmas
def check_dfs_order():
    explorer = ExhaustiveExplorer('../dse_database/poly/config/', dse_kernel, '../dse_database/programl/poly/processed', {})
    ordered_pids = explorer.ordered_pids
    ordered_pids.reverse()
    print(f'DFS exploration has {len(ordered_pids)} pragmas. Order:', ordered_pids)
    return ordered_pids


def check_autodse_best_point_pragma_order():
    dataset = SimpleDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    key_list = []
    perf_list = []
    for _, data in enumerate(tqdm(loader)):
        config_str = data[0].key
        if dse_kernel not in check_kernels:    # some kernels have 'lv2:' at the begining of the string
            assert config_str[0:4] in ['lv1:', 'lv2:']
            config_str = config_str[4:]
        else:
            assert config_str[0:4] not in ['lv1:', 'lv2:']
        lut_mask = (data[0]['util_LUT'] <= 0.8)
        ff_mask = (data[0]['util_FF'] <= 0.8)
        dsp_mask = (data[0]['util_DSP'] <= 0.8)
        bram_mask = (data[0]['util_BRAM'] <= 0.8)
        mask = lut_mask * ff_mask * dsp_mask * bram_mask
        if mask:
            key_list.append(config_str)
            perf_list.append(data[0].actual_perf)
    best_idx = np.argmin(perf_list)
    print('AutoDSE best latency:', perf_list[best_idx])
    best_key = key_list[best_idx]
    print('AutoDSE best design:', best_key)
    best_point = dse_utils.str_to_point(best_key)

    config_path = dse_utils.get_config_path('poly', dse_kernel)
    # config_path: ../dse_database/poly/config/..._ds_config.json
    ds_config = dse_utils.load_config(config_path, saver)
    ds, num_ds = compile_design_space(ds_config['design-space']['definition'], None, saver)
    default_point = get_default_point(ds)
    dfs_order = check_dfs_order()
    diff_keys = []
    print('Differences between AutoDSE best point and the default point:')
    for key, value in default_point.items():
        if best_point[key] != value:
            print(f'{key}: default is {value}, the AutoDSE value is {best_point[key]}')
            diff_keys.append(key)
    
    diff_key_order = [dfs_order.index(k) + 1 for k in diff_keys]
    print(diff_key_order)
    print(np.mean(diff_key_order))


def check_autodse_best_point_dfs_order():
    key_list = []
    perf_list = []
    file_cnt = 0
    for _, file in enumerate(tqdm(os.listdir(SAVE_DIR))):
        if file in ['raw', 'processed', 'encoders.klepto', 'pragma_dim.klepto']:
            continue
        data = torch.load(f'{SAVE_DIR}/{file}', map_location='cpu')
        file_cnt += 1
        config_str = data.key
        if dse_kernel not in check_kernels:    # some kernels have 'lv2:' at the begining of the string
            assert config_str[0:4] in ['lv1:', 'lv2:']
            config_str = config_str[4:]
        else:
            assert config_str[0:4] not in ['lv1:', 'lv2:']
        lut_mask = (data['util_LUT'] <= 0.8)
        ff_mask = (data['util_FF'] <= 0.8)
        dsp_mask = (data['util_DSP'] <= 0.8)
        bram_mask = (data['util_BRAM'] <= 0.8)
        mask = lut_mask * ff_mask * dsp_mask * bram_mask
        if mask:
            key_list.append(config_str)
            perf_list.append(data.actual_perf)
    best_idx = np.argmin(perf_list)
    print('AutoDSE best latency:', perf_list[best_idx])
    best_key = key_list[best_idx]
    print('AutoDSE best design:', best_key)
    best_point = dse_utils.str_to_point(best_key)
    print('AutoDSE data size:', file_cnt)

    explorer = ExhaustiveExplorer('../dse_database/poly/config/', dse_kernel, '../dse_database/programl/poly/processed', {})
    explored_len = 0
    tqdm_bar = tqdm()
    
    for batch in explorer.gen():
        for batch_i, point in enumerate(batch):
            equal = True
            for pragma in best_point.keys():
                if point[pragma] != best_point[pragma]:
                    equal = False
                    break
            if equal:
                explored_len += batch_i + 1
                break
        if equal:
            break
        tqdm_bar.update(len(batch))
        explored_len += len(batch)
    print(f"After DFS searching for {explored_len} points, we reached the AutoDSE's best design for {dse_kernel}")


def check_autodse_best_perf_for_all_finetune_kernels():
    TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
    SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'{FLAGS.v_db}_MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}', "finetune_dataset")
    all_results = {}
    existing_files = os.listdir(SAVE_DIR)
    for file in tqdm(existing_files):
        if file in ['raw', 'processed', 'encoders.klepto', 'pragma_dim.klepto']:
            continue
        data = torch.load(f'{SAVE_DIR}/{file}')
        gname = data['gname']
        if data['util_LUT'] > 0.8 or data['util_FF'] > 0.8 or data['util_DSP'] > 0.8 or data['util_BRAM'] > 0.8:
            continue
        if gname in all_results.keys():
            all_results[gname][data['key']] = data['actual_perf'].item()
        else:
            all_results[gname] = {data['key']: data['actual_perf'].item()}
    
    for gname, results in all_results.items():
        key = list(results.keys())[0]
        no_pragma_key = ''
        pragmas = key.split('.')
        for pragma in pragmas:
            pragma = pragma.split('-')
            if pragma[1] in ['NA', 'off', 'cg', 'flatten', 'normal', 'reduction']:
                no_pragma_key += f'{pragma[0]}-off.'
            else:
                number = int(pragma[1])
                no_pragma_key += f'{pragma[0]}-1.'
        no_pragma_key = no_pragma_key[:-1]
        print(f'{gname} min latency: {min(results.values())}, max latency: {max(results.values())}, '
            f'mean latency: {np.mean(list(results.values())):.0f}+-{np.std(list(results.values())):.0f}, '
            f'no pragma latency: {results[no_pragma_key]}')


if __name__ == '__main__':
    # calculate_overlap_perf(plot=False, search_limit=1e6)
    check_autodse_best_point_pragma_order()
    # check_autodse_best_point_dfs_order()
    # check_autodse_best_perf_for_all_finetune_kernels()
