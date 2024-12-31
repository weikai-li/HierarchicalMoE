import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from parameter import DesignSpace, DesignPoint
from result import Result
from dse import GNNModel
from config import FLAGS
from RL import dse_utils

import copy, torch_geometric
import networkx as nx
from typing import List, Dict, Union, Any, Tuple
from tqdm import tqdm


class ActionSpace:
    def __init__(self, ordered_pids: List[str], full_action_map: Dict[int, List[Union[str, int]]]):
        self.ordered_pids = ordered_pids
        self.pids_to_idx = dict()
        for i, pid in enumerate(self.ordered_pids):
            self.pids_to_idx[pid] = i
        self.full_action_map = full_action_map

    def to_idx_offset(self, pid: str, value: Union[str, int]):
        idx = self.pids_to_idx[pid]
        offset = self.full_action_map[idx].index(value)
        return idx, offset

    def from_ds(ds: DesignSpace, ordered_pids: List[str]):
        full_action_map = dict()
        for idx, pid in enumerate(ordered_pids):
            expr = copy.copy(ds[pid].option_expr)
            t = expr.find('if')
            if t != -1:
                #FIXME: handle more robust cases?
                expr = expr[:t]
                expr += ']'
            options = eval(expr)     # eval() transforms the string expr to a list
            full_action_map[idx] = options
        return ActionSpace(ordered_pids, full_action_map)
    
    
# used as a namespace
class RLZoo:
    def clone_point(point: DesignPoint):
        return dict(point)
    
    # Use the loaded model to test the given points. If the classification model predicts validity,
    # but the resources predicted by the regression model exceed the limit, we still add the regression model's results
    def get_results(
        graph: nx.Graph, config: Dict[str, Any], reg_model: GNNModel, class_model: GNNModel, points: List[DesignPoint], 
        state_type: str, have_uncertainty=False, uncertainty_est=None, uncertainty_bound: int = None,
        kernel: Tuple[str] = None, return_datalist = False, class_datalist = None, reg_datalist = None
    ):
        data_list = class_datalist
        if data_list is None:
            data_list = []
            for point in points:
                data_list.append(dse_utils.apply_design_point(graph, point, 'class',
                    kernel=kernel, no_point=False).to('cpu'))
        test_loader = torch_geometric.data.DataLoader(data_list, batch_size=FLAGS.batch_size, shuffle=False)
        rets = [None for i in range(len(points))]
        valids, class_embeds = class_model.test(
            test_loader, config['evaluate'], mode='class', return_embed=True
        )
        for i in range(len(points)):
            if valids[i] == 0:
                new_res = Result('UNAVAILABLE')
                new_res.point = data_list[i].point
                if state_type == 'None':
                    rets[i] = new_res
                else:
                    rets[i] = tuple([new_res, class_embeds[i]])
        
        reg_data_list = reg_datalist
        if reg_data_list is None:
            reg_data_list = []
            for idx, point in enumerate(points):
                reg_data_list.append(dse_utils.apply_design_point(graph, point, 'regression',
                    kernel=kernel, no_point=False).to('cpu'))
        test_loader = torch_geometric.data.DataLoader(reg_data_list, batch_size=FLAGS.batch_size, shuffle=False)
        results, embeds = reg_model.test(
            test_loader, config['evaluate'], mode='regression', return_embed=True
        )

        for i in range(len(points)):
            if rets[i] is None:   # In this case, this point is predicted to be valid
                if have_uncertainty:
                    _v = uncertainty_est.estimate(embeds[i]).squeeze().cpu().numpy()
                    if _v > uncertainty_bound:
                        results[i].valid = False
                    results[i].reg_uncertainty = _v
                if state_type == 'embed':
                    results[i].point = reg_data_list[i].point
                    rets[i] = tuple([results[i], embeds[i]])
                elif state_type == 'None':
                    results[i].point = reg_data_list[i].point
                    rets[i] = results[i]
                else:
                    raise NotImplementedError()
        if return_datalist:
            return rets, reg_data_list, data_list
        return rets
