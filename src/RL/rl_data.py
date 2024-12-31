import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dse import GNNModel
from RL import dse_utils
from parameter import DesignPoint
from config import FLAGS

from torch_geometric.data import DataLoader
import torch
import numpy as np
import networkx as nx


class PrivateDB:
    def __init__(
        self, g: nx.Graph, reg_model: GNNModel, class_model: GNNModel, kernel=None, redis_port=None
    ):
        self.G = g
        self.reg_model = reg_model; self.class_model = class_model
        self.kernel = kernel
        self.reg_data_list = []
        self.class_data_list = []
        self.key_list = []
        self.redis_port = redis_port
    
    def extend(self, new_db_path: str, rets=None):
        if new_db_path != None:
            assert rets == None
            self.reg_data_list = self.reg_data_list + dse_utils.get_datalist(
                new_db_path, self.G, 'regression', redis_port=self.redis_port
            )
            new_class_list, new_key_dict = dse_utils.get_datalist(
                new_db_path, self.G, 'class', True, redis_port=self.redis_port
            )
        else:
            assert rets != None
            self.reg_data_list = self.reg_data_list + dse_utils.get_datalist(
                None, self.G, 'regression', redis_port=self.redis_port, res_list=rets
            )
            new_class_list, new_key_dict = dse_utils.get_datalist(
                None, self.G, 'class', True, redis_port=self.redis_port, res_list=rets
            )
        self.class_data_list = self.class_data_list + new_class_list

        _min_best_perf = float('inf')
        chosen = None
        self.key_list.extend(list(new_key_dict.keys()))
        for res in new_key_dict.values():
            if res.valid and res.perf < _min_best_perf:
                _min_best_perf = res.perf
                chosen = res
        return chosen
    
    def get_loader(self, shuffle = True, batch_size = FLAGS.batch_size):
        reg_loader, class_loader = None, None
        if len(self.reg_data_list) != 0:
            reg_loader = DataLoader(self.reg_data_list, batch_size=batch_size, shuffle=shuffle)
        if len(self.class_data_list) != 0:
            class_loader = DataLoader(self.class_data_list, batch_size=batch_size, shuffle=shuffle)
        return reg_loader, class_loader

    def is_exist(self, point: DesignPoint):
        return dse_utils.point_to_str(point) in self.key_list
