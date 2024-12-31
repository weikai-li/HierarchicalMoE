from typing import List, Dict, Tuple, Any, Union
import numpy as np
from tqdm import tqdm
import random, uuid
import os, time, torch, pickle
from torch_geometric.loader import DataLoader

from parameter import DesignPoint, DesignSpace, compile_design_space, topo_sort_param_ids
from result import Result
from dse import ExhaustiveExplorer
from data import create_edge_index
from utils import get_root_path
from saver import saver
from config import FLAGS
from model import Net, HierarchicalMoE
import RL.dse_utils as dse_utils
from RL.dse_utils import point_to_str
from RL.hls import PrivateHQ


def uniform_sample_one(ds: DesignSpace, perturb: int, sv_cnt: Dict[Tuple[str, int], int]):
    if sv_cnt == None:
        cnt: Dict[Tuple[str, int], int] = dict()
    else:
        cnt = sv_cnt
    
    def _count(ds: DesignSpace, sorted_pids: List[str], h: int, point: DesignPoint):
        cur_state = tuple([point_to_str(point), h])
        if cur_state in cnt:
            return cnt[cur_state]
        if h == len(sorted_pids):
            cnt[cur_state] = 1
            return 1
        pid = sorted_pids[h]
        param = ds[pid]
        options = eval(param.option_expr, point.copy())
        cnt[cur_state] = 0
        if param.child:
            for option in options:
                point[pid] = option # ch
                cnt[cur_state] += _count(ds, sorted_pids, h+1, point)
            point.pop(pid)
            assert cur_state[0] == point_to_str(point)
        else:
            assert cur_state[0] == point_to_str(point)
            cnt[cur_state] = len(options) * _count(ds, sorted_pids, h+1, point)
        return cnt[cur_state]
    
    def dfs(ds: DesignSpace, sorted_pids: List[str], h: int, point: DesignPoint):
        if h == len(sorted_pids):
            return point.copy()
        pid = sorted_pids[h]
        param = ds[pid]
        options = eval(param.option_expr, point.copy())
        ret = None
        if h < perturb:
            # uniform probability for each point
            _id = random.randint(0, len(options)-1)
            if param.child:
                point[pid] = options[_id]
            ret = dfs(ds, sorted_pids, h+1, point)
            if not param.child:
                ret[pid] = options[_id]
        else:
            if param.child:
                cnt_leaf = []
                for option in options:
                    point[pid] = option
                    cnt_leaf.append(_count(ds, sorted_pids, h+1, point))
                _sum = 0.0
                for cnt in cnt_leaf:
                    _sum += cnt
                for i in range(len(cnt_leaf)):
                    cnt_leaf[i] /= _sum
                _id = np.random.choice(np.arange(len(options), dtype=int), p=cnt_leaf)
                point[pid] = options[_id]
                ret = dfs(ds, sorted_pids, h+1, point)
            else:
                _id = random.randint(0, len(options)-1)
                ret = dfs(ds, sorted_pids, h+1, point)
                ret[pid] = options[_id]
        if pid in point:
            point.pop(pid)
        return ret
    sorted_pids = topo_sort_param_ids(ds)
    
    point: Dict[str, Any] = dict()
    _count(ds, sorted_pids, 0, point)
    ret = dfs(ds, sorted_pids, 0, point)
    return ret, cnt


def random_sample(ds: DesignSpace, N: int, show_tqdm=True):
    ret: List[DesignPoint] = []
    cnt = None
    # print('Init Leaf count...')
    # point, cnt = uniform_sample_one(ds, perturb=0, sv_cnt=cnt)
    range_bar = range(N)
    if show_tqdm == True:
        range_bar = tqdm(range_bar)
    for _ in range_bar:
        # could be partially uniform
        point, cnt = uniform_sample_one(ds, perturb=0, sv_cnt=cnt)
        ret.append(point)
    return ret


def exhaustive_search(dataset, dse_kernel, dse_round, search_limit, search_time_limit, dse_hls_num):
    config_path = dse_utils.get_config_path(dataset, dse_kernel)
    # config_path: /home/username/software-gnn/dse_database/poly/config/..._ds_config.json
    ds_config = dse_utils.load_config(config_path, saver)
    ds, num_ds = compile_design_space(ds_config['design-space']['definition'], None, saver)
    # sampled_points = random_sample(ds, 50)
    # print(sampled_points)

    # explorer = Explorer('../dse_database/poly/config/', dse_kernel, '../dse_database/programl/poly/processed')
    explorer = ExhaustiveExplorer(f'../dse_database/{dataset}/config/', dse_kernel, f'../dse_database/programl/{dataset}/processed', {})
    explored_len = 0

    # Finetuned regression model
    model_path = f'{get_root_path()}/src/logs/auto-encoder/iccad/round1-40kernel/pragma_as_MLP/parallel_and_merge/class/post-gnn-3lp-3lm/dropout-0.1-MSE-scheduler-warmup-weight-decay-0.0001-no-mutual-info-PB-edge-attr-True_position-True-6L-SSL-False-gae-T-False-gae-P-False_regression_train_2024-10-15T00-02-41.981646/run1/1000_val_model_state_dict.pth'
    # Finetuned class model
    class_model_path = f'{get_root_path()}/src/logs/auto-encoder/iccad/round1-40kernel/pragma_as_MLP/parallel_and_merge/class/post-gnn-3lp-3lm/dropout-0.1-MSE-scheduler-warmup-weight-decay-0.0001-no-mutual-info-PB-edge-attr-True_position-True-6L-SSL-False-gae-T-False-gae-P-False_class_train_2024-10-15T10-40-30.183471/run1/500_val_model_state_dict.pth'

    if len(FLAGS.moe_layers) > 0 and FLAGS.moe_layers[0][0:9] == 'hierarchy':
        assert len(FLAGS.moe_layers) == 1
        model = HierarchicalMoE(153, edge_dim=335, task='regression').to(FLAGS.device)
    else:
        model = Net(153, edge_dim=335, task='regression').to(FLAGS.device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    class_model = HierarchicalMoE(153, edge_dim=335, task='class').to(FLAGS.device)
    # class_model = Net(153, edge_dim=335, task='class', no_moe=True).to(FLAGS.device)
    class_model.load_state_dict(torch.load(class_model_path, map_location=torch.device('cpu')))
    class_model.eval()
    
    if dse_round != 0:   # Normal DSE, not random
        if dse_round == 1 or dse_round == 10:
            explored_idx = []
        else:
            explored_idx = []
            for round in range(1, dse_round):
                if len(FLAGS.moe_layers) == 0:
                    tmp_idx = np.load(f'../hls_result/{dse_kernel}/moe_no/{round}_explored_idx.npy')
                else:
                    tmp_idx = np.load(f'../hls_result/{dse_kernel}/moe_{FLAGS.moe_layers[0]}/{round}_explored_idx.npy')
                explored_idx.extend(tmp_idx)
                
        best_points = []
        tqdm_bar = tqdm(search_limit)
        start_time = time.time()
        edge_attr = explorer.GNNmodel.encode_edge(explorer.graph)
        edge_index = create_edge_index(explorer.graph, FLAGS.device)
        for batch in explorer.gen():
            explored_time = time.time() - start_time
            if explored_len > search_limit or explored_time > search_time_limit * 60:
                break
            data_list = []
            for point in batch:
                data_list.append(explorer.apply_design_point(explorer.graph, point, mode='regression',
                                                             model=None, edge_attr=edge_attr, edge_index=edge_index))
            test_loader = DataLoader(data_list, batch_size=FLAGS.batch_size)
            best_in_batch = []
            with torch.no_grad():
                for data in test_loader:
                    data.to(FLAGS.device)
                    out_dict, loss, loss_dict_, gae_loss = model(data, test_mode=True)
                    class_out_dict, _, _, _ = class_model(data, test_mode=True)
                    valid_mask = (class_out_dict['perf'][:, 1] > class_out_dict['perf'][:, 0]).type(torch.uint8).unsqueeze(-1)
                    lut_mask = (out_dict['util-LUT'] < 0.8).type(torch.uint8)
                    ff_mask = (out_dict['util-FF'] < 0.8).type(torch.uint8)
                    dsp_mask = (out_dict['util-DSP'] < 0.8).type(torch.uint8)
                    bram_mask = (out_dict['util-BRAM'] < 0.8).type(torch.uint8)
                    mask = valid_mask * lut_mask * ff_mask * dsp_mask * bram_mask
                    out_dict['perf'] *= mask
            for i in range(len(batch)):
                if i + explored_len not in explored_idx:
                    best_in_batch.append((out_dict['perf'][i].item(), batch[i], i + explored_len))
            combined = best_points + best_in_batch
            best_points = sorted(combined, key=lambda x: -x[0])[:dse_hls_num]
            explored_len += len(batch)
            tqdm_bar.update(len(batch))
        best_data = [i[1] for i in best_points]
        best_data_idx = [i[2] for i in best_points]
        print('best_data_idx:', best_data_idx)
    
    else:
        assert dse_round == 0   # random
        val_point_list = []
        tqdm_bar = tqdm(search_limit)
        edge_attr = explorer.GNNmodel.encode_edge(explorer.graph)
        edge_index = create_edge_index(explorer.graph, FLAGS.device)
        for batch in explorer.gen():
            if explored_len > search_limit:
                break
            data_list = []
            for point in batch:
                data_list.append(explorer.apply_design_point(explorer.graph, point, mode='regression',
                                                             model=None, edge_attr=edge_attr, edge_index=edge_index))
            test_loader = DataLoader(data_list, batch_size=FLAGS.batch_size)
            for data in test_loader:
                data = data.to(FLAGS.device)
                class_out_dict, _, _, _ = class_model(data)
                valid_mask = (class_out_dict['perf'][:, 1] > class_out_dict['perf'][:, 0])
            valid_idx = torch.nonzero(valid_mask).squeeze(1)
            for i in valid_idx:
                val_point_list.append(batch[i])
            explored_len += len(batch)
            tqdm_bar.update(len(batch))
            if explored_len > 10000:
                random.shuffle(val_point_list)
                val_point_list = val_point_list[:300]
        random.shuffle(val_point_list)
        best_data = val_point_list[:300]
    return best_data


def save_best_data(best_data, uid):
    print('uid:', uid)
    os.makedirs(f'../hls_local/{uid}')
    torch.save(best_data, f'../hls_local/{uid}/best_data.pt')


def main(load_pickle=False, load_pt=False):
    assert load_pickle + load_pt in [0, 1]

    dse_kernel = FLAGS.dse_kernel
    dse_round = 1   # dse_round = 0 means random, 10 means last time top10, otherwise is normal
    dse_hls_num = 10
    search_limit = 75000
    search_time_limit = 60   # min
    MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack',
        'stencil', 'nw', 'md', 'stencil-3d']
    poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen',
        'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
        'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
        'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
        'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium',
        'gemver-medium']
    if dse_kernel in MACHSUITE_KERNEL:
        benchmark = 'machsuite'
    else:
        assert dse_kernel in poly_KERNEL
        benchmark = 'poly'
 
    if load_pt == True:
        uid = FLAGS.uid
        best_data = torch.load(f'../hls_local/{uid}/{FLAGS.dse_kernel}.pt')
    
    elif load_pickle == True:
        best_data = []
        pkl_path = f'../hls_local/0501_atefeh_harp/{dse_kernel}_cpu.pkl'
        with open(pkl_path, 'rb') as f:
            L = pickle.load(f)
            for line in L:
                new_point = dict()
                eles = line.split('.')
                for ele in eles:
                    vs = ele.split('-')
                    if 'PARA' in vs[0] or 'TILE' in vs[0]:
                        vs[1] = int(vs[1])
                    else:
                        if vs[1] == 'NA':
                            vs[1] = ''
                    new_point[vs[0]] = vs[1]
                best_data.append(new_point)
        uid = uuid.uuid4()
        save_best_data(best_data, uid)
    
    else:
        best_data = exhaustive_search(benchmark, dse_kernel, dse_round, search_limit, search_time_limit, dse_hls_num)
        uid = uuid.uuid4()
        save_best_data(best_data, uid)

    redis_port = FLAGS.redis_port
    hq = PrivateHQ(dse_hls_num, benchmark, dse_kernel, uid, 0, redis_port=redis_port)
    for point in best_data:
        new_res = Result()
        new_res.valid = True
        new_res.perf = 1.0
        new_res.point = point
        hq.append(new_res)
    db_path = hq.query_batch_remote(timeout=180)   # time limit is 180 min
    print(db_path)

    explorer = ExhaustiveExplorer(f'../dse_database/{benchmark}/config/', dse_kernel,
                                  f'../dse_database/programl/{benchmark}/processed', {})
    reg_datalist = dse_utils.get_datalist(db_path, explorer.graph, 'regression', no_point=True,
                                          kern=dse_kernel, redis_port=redis_port)
    class_datalist = dse_utils.get_datalist(db_path, explorer.graph, 'class', no_point=True,
                                            kern=dse_kernel, redis_port=redis_port)

    print(f'{len(reg_datalist)} regression results')
    print(f'{len(class_datalist)} classification results')


if __name__ == '__main__':
    main()
