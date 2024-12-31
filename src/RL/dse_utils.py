# dse_utils works below dse.py
import sys
import os
import psutil
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameter import DesignPoint
from data import _encode_X_torch, _encode_edge_torch, create_edge_index, _encode_X_dict, _encode_edge_dict
from dse import GNNModel, SAVE_DIR
from saver import saver
from config import FLAGS
from utils import get_root_path, load
from config_ds import build_config

from typing import List
from torch_geometric.data import Data
from typing import Any, Dict, List, Tuple
import numpy as np
import networkx as nx
import pickle, io, math, torch, redis, json
from glob import glob, iglob
import random


def get_pragmas(point: DesignPoint) -> List[int]:
    pragmas = []
    for _, value in sorted(point.items()):
        if type(value) is str:
            if value.lower() == 'flatten':
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
    if FLAGS.pragma_uniform_encoder:
        pragmas.extend([0] * (21 - len(pragmas)))
    return pragmas


def geo_data(
    mode: str, d_node: Dict[str, Any], point: DesignPoint, X: torch.FloatTensor, 
    edge_index: torch.Tensor, edge_attr: Any, no_point: bool = False, kernel: Tuple[str] = None
):
    if 'regression' in mode:
        data = Data(
            X_contextnids=d_node['X_contextnids'],
            X_pragmanids=d_node['X_pragmanids'],
            X_pragmascopenids=d_node['X_pragmascopenids'],
            X_pseudonids=d_node['X_pseudonids'],
            X_icmpnids=d_node['X_icmpnids'],
            X_pragma_per_node=d_node['X_pragma_per_node'],
            x=X,
            dataset = None if kernel is None else kernel[0],
            kernel = None if kernel is None else kernel[1],
            edge_index=edge_index,
            pragmas=d_node['pragmas'],
            perf=d_node['perf'],
            actual_perf=d_node['actual_perf'],
            quality=d_node['quality'],
            util_BRAM=d_node['util-BRAM'],
            util_DSP=d_node['util-DSP'],
            util_LUT=d_node['util-LUT'],
            util_FF=d_node['util-FF'],
            total_BRAM=d_node['total-BRAM'],
            total_DSP=d_node['total-DSP'],
            total_LUT=d_node['total-LUT'],
            total_FF=d_node['total-FF'],
            point=point,
            key=str(point),
            edge_attr=edge_attr
        )
    elif 'class' in mode:
        data = Data(
            x=X,
            edge_index=edge_index,
            pragmas=d_node['pragmas'],
            perf=d_node['perf'],
            point=point,
            key=str(point),
            edge_attr = edge_attr,
            dataset = None if kernel is None else kernel[0],
            kernel = None if kernel is None else kernel[1],
            X_contextnids=d_node['X_contextnids'],
            X_pragmanids=d_node['X_pragmanids'],                    
            X_pragmascopenids=d_node['X_pragmascopenids'],                    
            X_pseudonids=d_node['X_pseudonids'],    
            X_icmpnids=d_node['X_icmpnids'],    
            X_pragma_per_node=d_node['X_pragma_per_node']
        )
    else:
        raise NotImplementedError()
    if no_point:
        delattr(data, 'point')
    return data
        
    
def apply_design_point(
    g: nx.Graph, point: DesignPoint, mode: str, kernel: Tuple[str] = None,
    no_point: bool = False, force_keep_pragma_attribute=False
) -> Data:
    encoder_path = FLAGS.encoder_path
    encoder = load(encoder_path, print_msg=False)
    X, d_node = encode_node(g, point, encoder, force_keep_pragma_attribute=force_keep_pragma_attribute)
    edge_attr = encode_edge(g, encoder)
    edge_index = create_edge_index(g, FLAGS.device)
    pragmas = get_pragmas(point)

    resources = ['BRAM', 'DSP', 'LUT', 'FF']
    keys = ['perf', 'actual_perf', 'quality']
    d_node['pragmas'] = torch.FloatTensor(np.array([pragmas]))
    for r in resources:
        keys.append('util-' + r)
        keys.append('total-' + r)
    for key in keys:
        d_node[key] = 0
    if mode == 'class':
        d_node['perf'] = 1
    return geo_data(mode, d_node, point, X, edge_index, edge_attr, kernel=kernel, no_point=no_point)


def get_model(pragma_dim, device, task, model_path=None) -> GNNModel:
    return GNNModel(
        SAVE_DIR, saver, task=task, pragma_dim=pragma_dim, device=device, model_path=model_path
    )


def encode_node(g, point: DesignPoint, encoder, force_keep_pragma_attribute: bool=False): ## prograML graph
    required_keys = ['X_contextnids', 'X_pragmanids', 'X_pseudonids', 'X_icmpnids', 'X_pragmascopenids', 'X_pragma_per_node']
    node_dict = _encode_X_dict(g, point=point, device=FLAGS.device, force_keep_pragma_attribute=force_keep_pragma_attribute)
    enc_ntype = encoder['enc_ntype']
    enc_ptype = encoder['enc_ptype']
    enc_itype = encoder['enc_itype']
    enc_ftype = encoder['enc_ftype']
    enc_btype = encoder['enc_btype']
    return _encode_X_torch(node_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype, FLAGS.device), \
        {k: node_dict[k] for k in required_keys}
    
    
def encode_edge(g, encoder):
    edge_dict = _encode_edge_dict(g)
    enc_ptype_edge = encoder['enc_ptype_edge']
    enc_ftype_edge = encoder['enc_ftype_edge']
    return _encode_edge_torch(edge_dict, enc_ftype_edge, enc_ptype_edge, FLAGS.device)


def get_datalist(
    db_path, g: nx.Graph, mode: str, return_key: bool=False,
    no_point=False, kern=None, res_list=None, redis_port=None
):
    if res_list == None:
        assert db_path != None
        database = redis.StrictRedis(host='localhost', port=redis_port, db=14)
        database.flushdb()
        res_list = []
        used_keys = []
        with open(db_path, 'rb') as handle:
            new_data = pickle.load(handle)
            database.hmset(0, new_data)
            keys = [k.decode('utf-8') for k in database.hkeys(0)]
            min_perf = float('inf')
            chosen = None
            for key in sorted(keys):
                pickle_obj = database.hget(0, key)
                # obj = pickle.Unpickler(io.BytesIO(pickle_obj)).load()
                obj = pickle.loads(pickle_obj)   # equivalent to the obove line
                if type(obj) is dict or type(obj) is int:   # This obj might be like {'tool_version': 'Vitis-21.1'}
                    continue
                try:
                    assert obj.point != {}
                except:
                    print(obj)
                    exit()
                if mode == 'regression':
                    if key[0:3] == 'lv1':
                        continue     # lv1 means early rejected by Merlin
                    if not FLAGS.invalid and obj.perf < FLAGS.min_allowed_latency:
                        continue     # They are invalid points
                    assert obj.perf != 0
                    if obj.perf < min_perf and obj.valid:   # obj.valid is the resource limit
                        min_perf = obj.perf
                        chosen = obj
                res_list.append(obj)
                used_keys.append(key)
            if mode == 'regression':
                print('Min:', min_perf)
                if chosen is not None and min_perf != float('inf'):
                    # print(chosen.point)
                    print(chosen.res_util)
        print(f'Loading {len(res_list)} points from {db_path}')
    
    else:
        assert db_path == None
        new_res_list = []
        used_keys = []
        min_perf = float('inf')
        chosen = None
        for obj in res_list:
            assert obj.point != {}
            if mode == 'regression':
                if not FLAGS.invalid and obj.perf < FLAGS.min_allowed_latency:   # invalid
                    continue
                assert obj.perf != 0
                if obj.perf < min_perf and obj.valid:   # obj.valid is the resource limit
                    min_perf = obj.perf
                    chosen = obj
            used_keys.append('lv2-key')
            new_res_list.append(obj)
        if mode == 'regression':
            print('Min:', min_perf)
            if chosen is not None and min_perf != float('inf'):
                print(chosen.point)
                print(chosen.res_util)
        res_list = new_res_list
    
    data_list = []
    encoder_path = FLAGS.encoder_path
    encoder = load(encoder_path, print_msg=False)
    for idx, res in enumerate(res_list):
        point = res.point
        X, d_node = encode_node(g, point, encoder)
        edge_attr = encode_edge(g, encoder)
        edge_index = create_edge_index(g, FLAGS.device)
        pragmas = get_pragmas(point)
        resources = ['BRAM', 'DSP', 'LUT', 'FF']
        targets = ['perf', 'actual_perf', 'quality']
        d_node['pragmas'] = torch.FloatTensor(np.array([pragmas])).half()
        for r in resources:
            targets.append('util-' + r)
            targets.append('total-' + r)
        if 'regression' in mode:
            for t in targets:
                y = None
                if t == 'perf':
                    if FLAGS.norm_method == 'speedup-log2':
                        y = math.log2(FLAGS.normalizer / res.perf) / 2
                    else:
                        raise NotImplementedError()
                elif t == 'quality':
                    y = 0
                elif t == 'actual_perf':
                    y = res.perf
                elif 'util' in t or 'total' in t:
                    y = res.res_util[t] * FLAGS.util_normalizer
                else:
                    raise NotImplementedError()
                if t != 'actual_perf' and t != 'quality' and (not 'total' in t):
                    d_node[t] = torch.FloatTensor(np.array([y]))
                else:
                    d_node[t] = torch.FloatTensor(np.array([y]))
        elif 'class' in mode:
            key = used_keys[idx]
            if 'lv1' in key:   # early rejected by Merlin
                lv2_key = key.replace('lv1', 'lv2')
                if lv2_key in used_keys:
                    continue
                else:
                    y = 0
            else:
                y = 0 if res.perf < FLAGS.min_allowed_latency else 1
            for t in targets:
                d_node[t] = 0
            d_node['perf'] = torch.FloatTensor(np.array([y])).type(torch.LongTensor)
        else:
            raise NotImplementedError()
        data_list.append(geo_data(mode, d_node, point, X, edge_index, edge_attr, no_point))
    
    if return_key:
        _ret_dict = dict()
        for i in range(len(res_list)):
            _ret_dict[point_to_str(res_list[i].point)] = res_list[i]
        return data_list, _ret_dict
    return data_list


def check_memory():
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    rss = memory_info.rss
    for unit in ['', 'K', 'M', 'G', 'T']:
        if rss < 1024:
            return f"{rss:.2f} {unit}B"
        rss /= 1024
    return f"{rss:.2f} PB"


def point_to_str(point: DesignPoint):
    ret = ''
    for k in sorted(point.keys()):
        ret += f'_{k}_{point[k]}'
    return ret

def str_to_point(key: str):
    point = {}
    pragmas = key.split('.')
    for pragma in pragmas:
        pragma = pragma.split('-')
        if pragma[1] == 'NA':
            point[pragma[0]] = ''
        else:
            try:
                point[pragma[0]] = int(pragma[1])
            except:
                point[pragma[0]] = pragma[1]
    return point


def get_config_path(dataset: str, kernel: str):
    return f'{get_root_path()}/dse_database/{dataset}/config/{kernel}_ds_config.json'
    
def load_config(config_path: str, logger, silent = False) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        print('Config JSON file not found: %s', config_path)
        raise RuntimeError()

    if not silent: print('Loading configurations')
    with open(config_path, 'r', errors='replace') as filep:
        try:
            user_config = json.load(filep)
        except ValueError as err:
            print('Failed to load config: %s', str(err))
            raise RuntimeError()

    config = build_config(user_config, logger)
    if config is None:
        print('Config %s is invalid', config_path)
        raise RuntimeError()
    
    return config


def get_graph_path(dataset: str):
    return os.path.join(get_root_path(), 'dse_database', 'programl', dataset, 'processed')

def get_graph(path_graph: str, kernel_name: str, silent=False) -> nx.Graph:
    if FLAGS.graph_type == '':
        pruner = 'extended'
        if FLAGS.use_for_nodes:
            gexf_file = sorted([f for f in iglob(path_graph + "/**/*", recursive=True) if f.endswith('.gexf') and f'{kernel_name}_' in f and pruner not in f and 'pseudo-for' in f])
        else:
            gexf_file = sorted([f for f in iglob(path_graph + "/**/*", recursive=True) if f.endswith('.gexf') and f'{kernel_name}_' in f and pruner not in f and 'pseudo-for' not in f])
    else:
        if FLAGS.use_for_nodes:
            gexf_file = sorted([f for f in glob(path_graph + "/**/*") if f.endswith('.gexf') and f'{kernel_name}_' in f and FLAGS.graph_type in f and 'pseudo-for' in f])
        else:
            gexf_file = sorted([f for f in glob(path_graph + "/**/*") if f.endswith('.gexf') and f'{kernel_name}_' in f and FLAGS.graph_type in f and 'pseudo-for' not in f])
    if not silent:
        print('Graph file path =', gexf_file)
    assert len(gexf_file) == 1
    graph_path = os.path.join(path_graph, gexf_file[0])
    return nx.read_gexf(graph_path)


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
