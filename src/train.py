from config import FLAGS
from saver import Saver, saver
from utils import MLP, OurTimer, MLP_multi_objective, plot_loss_trend, _get_y_with_target, create_dir_if_not_exists, plot_lr_trend
from data import get_kernel_samples, split_dataset, split_dataset_resample, split_train_test_kernel, MyOwnDataset
import data
SAVE_DIR = data.SAVE_DIR
from model import Net, HierarchicalMoE
if FLAGS.sample_finetune:
    import sample_finetune
    SAVE_DIR = sample_finetune.SAVE_DIR
import config
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL
from coreset import kmeans_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix

import os
import torch
import pytorch_warmup as warmup
from torch_geometric.data import DataLoader
import torch.nn as nn
from torch.utils.data import random_split
import shutil
import numpy as np
import random

from scipy.stats import rankdata, kendalltau

from tqdm import tqdm
from os.path import join, basename
import learn2learn as l2l

from collections import OrderedDict, defaultdict

import pandas as pd


def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data,_ in d['pred']]
    pred = [data for _,data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')

def _report_rmse_etc(points_dict, label, print_result=True):
    if print_result:
        saver.log_info(label)
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num_data = None
    try:
        for target_name, d in points_dict.items():
            # true_li = d['true']
            # pred_li = d['pred']
            true_li = [data for data,_ in d['pred']]
            pred_li = [data for _,data in d['pred']]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li, squared=False)
            mse = mean_squared_error(true_li, pred_li, squared=True)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)

            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)

            # data['rmse'].append(f'{rmse:.4f}')
            # data['mse'].append(f'{mse:.4f}')
            # data['tau'].append(f'{tau: .4f}')
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        saver.log_info(f'Error {v}')
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    # exit()
    return df
    # exit()

def feature_extract(model, key_word, gnn_layer=None):
    '''"
        fixes all parameters except for the ones that have "key_word" 
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                saver.log_info(f'fixing parameter: {name}')
                param.requires_grad = False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer-1)]):
                    saver.log_info(f'fixing parameter: {name}')
                    param.requires_grad = False
    
    if FLAGS.random_MLP:
        D = FLAGS.D
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]
        if FLAGS.node_attention:
            dim = FLAGS.separate_T + FLAGS.separate_P + FLAGS.separate_pseudo + FLAGS.separate_icmp
            in_D = dim * D
        else:
            in_D = D
        if model.MLP_version == 'single_obj':
            for target in FLAGS.target:
                model.MLPs[target] = MLP(in_D, 1, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels,
                                        num_hidden_lyr=len(hidden_channels))
        else:
            model.MLPs = MLP_multi_objective(in_D, 1, activation_type=FLAGS.activation,
                                    hidden_channels=hidden_channels,
                                    objectives=FLAGS.target,
                                    num_common_lyr=FLAGS.MLP_common_lyr)



def check_feature_extract(model, key_word, gnn_layer=None):
    '''"
        checks that all parameters except for the ones that have "key_word" are fixed
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                assert param.requires_grad == False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer-1)]):
                    assert param.requires_grad == False


def gen_dataset(li, batch_size=FLAGS.batch_size):
    # li[0] is MyOWnDataset
    if FLAGS.finetune == True and FLAGS.train_mode in ['save_hidden', 'save_moe', 'observe_moe_distribution']:
        train_loader = DataLoader(li[0], batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
    elif FLAGS.finetune == True and FLAGS.train_mode == "save_pred_for_contest":
        train_loader = DataLoader(li[0], batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    else:
        train_loader = DataLoader(li[0], batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(li[1], batch_size=batch_size, pin_memory=False, num_workers=0)  # TODO: split make sure no seen kernels in val/test
    test_loader = DataLoader(li[2], batch_size=batch_size, pin_memory=False, num_workers=0)  # TODO

    num_features = train_loader.dataset[0].num_features
    saver.info(f'num features for training: {num_features}')
    edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    saver.info(f'size of the edge attribute is {edge_dim}')

    return train_loader, val_loader, test_loader, num_features, edge_dim 

def process_split_data(dataset):
    dataset_dict = defaultdict(list)
    dataset_dict['train'] = dataset
    dataset_dict['test'] = None
    if not FLAGS.all_kernels:
        dataset = get_kernel_samples(dataset)
        dataset_dict['train'] = dataset
    elif FLAGS.test_kernels is not None:
        dataset_dict = split_train_test_kernel(dataset)
        
    return dataset_dict

def get_train_val_count(num_graphs, val_ratio, test_ratio):
    if FLAGS.sample_finetune:
        r1 = int(num_graphs * 1.0)
        r2 = int(num_graphs * 0)
    elif FLAGS.test_kernels is not None:
        r1 = int(num_graphs * (1.0 - val_ratio))
        r2 = int(num_graphs * (val_ratio))
    else:
        r1 = int(num_graphs * (1.0 - val_ratio - test_ratio))
        r2 = int(num_graphs * (val_ratio))
        
    return r1, r2

def inference(dataset, init_pragma_dict=None, model_path=FLAGS.model_path, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.test_ratio, resample=-1, model_id=0, is_train_set=False, is_val_set=False):
    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)
    if resample == -1:
        li = split_dataset(dataset_dict['train'], r1, r2, dataset_test=dataset_dict['test'])
    else:
        li = split_dataset_resample(dataset, 1.0 - val_ratio - test_ratio, val_ratio, test_ratio, test_id=resample)
    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    test_set = test_loader 
    if is_train_set: 
        test_set = train_loader
        saver.info('running inference on train set')
    elif is_val_set: 
        test_set = val_loader
        saver.info('running inference on val set')
    
    if init_pragma_dict is None:
        init_pragma_dict = {'all': [1, 21]}
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=init_pragma_dict).to(FLAGS.device)

    if model_path != None:
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        shutil.copy(model_path, join(saver.logdir, f"{(basename(model_path)).split('.')[0]}-{model_id}.pth"))
    else:
        saver.error(f'model path should be set during inference')
        raise RuntimeError()

    if model_id == 0:
        saver.log_model_architecture(model)
    data_list = []
    
    if FLAGS.task == 'regression':
        csv_dict = {'header' : ['gname', 'pragma']}
        test_loss, loss_dict, gae_loss, MSE_loss = test(test_set, 'test', model, 0, plot_test = True, csv_dict = csv_dict, data_list =data_list, is_train_set=is_train_set, is_val_set=is_val_set)
        loss_dict = {k: round(v, 4) for k, v in loss_dict.items()}
        saver.log_info((f'{loss_dict}'))
        saver.log_info(('Test loss: {:.7f}, MSE loss: {:.7f}'.format(test_loss, MSE_loss)))
        saver.log_dict_of_dicts_to_csv(f'actual-prediction-{model_id}', csv_dict, csv_dict['header'])
        print(len(data_list), 'out of', len(test_loader))
        if FLAGS.get_forgetting_examples:
            NEW_SAVE_DIR = SAVE_DIR.replace('round', 'forgetting-round')
            saver.log_info(f'Saving {len(data_list)} to disk {NEW_SAVE_DIR}; Deleting existing files')
    else:
        test_loss, loss_dict_test = test(test_loader, 'test', model, 0)
        saver.log_info(('Test loss: {:.3f}'.format(test_loss)))
        

def model_update(model, losses_list, loss, epoch, plot_test, tag, saver=saver):
    saver.writer.add_scalar(f'{tag}/{tag}', loss, epoch)
    if losses_list and loss < min(losses_list):
        if FLAGS.save_model:
            saver.log_info((f'Saved {tag} model at epoch {epoch}'))
            save_epoch = (int(epoch / 500) + 1) * 500
            torch.save(model.state_dict(), join(saver.model_logdir, f"{save_epoch}_{tag}_model_state_dict.pth"))
            print(join(saver.model_logdir, f"{save_epoch}_{tag}_model_state_dict.pth"))
        plot_test = True
    losses_list.append(loss)
        
    return plot_test

def log_loss(loss_dict, gae_loss, tag, saver):
    saver.log_info((f'{tag} GAE loss: {gae_loss}'))
    saver.log_info((f'{tag} loss breakdown {loss_dict}'))


def save_moe(train_loader, model):
    assert len(FLAGS.moe_layers) == 1
    FLAGS.moe_layers = [FLAGS.observe_moe_layer]
    group_ids = {}
    gate_list = []
    hidden_list = []
    gate_to_middle = {'gnn2': 1, 'gnn3': 2, 'gnn4': 3, 'gnn5': 4, 'gnn6': 5,
        'pragma_mlp': 6, 'output_mlp': 7, 'pseudo_alone_w/o_gnn7': 6.5, 'gnn7': 6.3,
        'hierarchy-weighted-hidden': 9, 'hierarchy-top-hidden': 9}
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            data = data.to(FLAGS.device)
            base_idx = i * train_loader.batch_size
            for j, kernel_name in enumerate(data['kernel']):
                kernel_name = kernel_name.split('_')[0]
                if kernel_name in group_ids.keys():
                    group_ids[kernel_name].append(base_idx + j)
                else:
                    group_ids[kernel_name] = [base_idx + j]
            hidden_idx = gate_to_middle[FLAGS.observe_moe_layer]
            gate = model(data, return_gate=FLAGS.observe_moe_layer)
            gate_list.append(gate)
            hidden = model(data, return_middle=hidden_idx)
            hidden_list.append(hidden)
            torch.cuda.empty_cache()
    
    gate_list = torch.concat(gate_list)
    hidden_list = torch.concat(hidden_list)
    print(gate_list.shape, hidden_list.shape)
    save_name = f'hidden_save/moe_{FLAGS.observe_moe_layer}'
    np.save(f'{save_name}_gate.npy', gate_list.cpu().numpy())
    np.save(f'{save_name}_hidden.npy', hidden_list.cpu().numpy())
    print(f'save: {save_name}_gate.npy and {save_name}_hidden.npy')
    if FLAGS.observe_moe_layer == 'output_mlp':
        save_name = 'hidden_save/kernel_index.npy'
        np.save(save_name, group_ids)


def observe_moe_distribution(train_loader, model):
    assert len(FLAGS.moe_layers) == 1 and isinstance(model, HierarchicalMoE)
    group_gates = {}
    model.eval()

    # for data in train_loader:
    #     assert data['X_contextnids'].sum() + data['X_pragmanids'].sum() + data['X_pseudonids'].sum() == len(data['x'])
    #     pragma_ids = data['X_pragma_per_node'].nonzero()[:, 0]
    #     pragma_ids = set([p.item() for p in pragma_ids])
    #     pragma_ids2 = data['X_pragmascopenids'].nonzero()
    #     pragma_ids2 = set([p.item() for p in pragma_ids2])
    #     assert pragma_ids == pragma_ids2
    #     pseudo_ids = data['X_pseudonids'].nonzero()
    #     pseudo_ids = set([p.item() for p in pseudo_ids])
    #     for p in pragma_ids:
    #         assert p in pseudo_ids
    # exit()

    if FLAGS.observe_moe_layer == 'hierarchy-weighted-hidden':
        with torch.no_grad():
            for data in tqdm(train_loader):
                data = data.to(FLAGS.device)
                gate = model(data, return_gate=FLAGS.observe_moe_layer)
                for j, kernel_name in enumerate(data['kernel']):
                    kernel_name = kernel_name.split('_')[0]
                    if kernel_name in group_gates.keys():
                        group_gates[kernel_name].append(gate[j])
                    else:
                        group_gates[kernel_name] = [gate[j]]
                torch.cuda.empty_cache()
        for k, gate in group_gates.items():
            gate = torch.stack(gate).mean(0)
            k = k.replace('-', '_')
            print(f'{k}_hier = [{gate[0]}, {gate[1]}, {gate[2]}]')
    
    elif FLAGS.observe_moe_layer in ['gnn7', 'pseudo_alone_w/o_gnn7']:
        with torch.no_grad():
            gate_x = []
            gate_context_nids, gate_pragmanids, gate_pragmascopenids, gate_pseudonids, gate_icmpnids, gate_nopragmanids = [], [], [], [], [], []
            gate_tile, gate_pipeline, gate_parallel1, gate_parallel2 = [], [], [], []
            gate_pipeline1, gate_pipeline5, gate_pipeline10 = [], [], []
            for data in tqdm(train_loader):
                data = data.to(FLAGS.device)
                gate = model(data, return_gate=FLAGS.observe_moe_layer)
                data_len = len(gate)
                if FLAGS.observe_moe_layer == 'gnn7':
                    # gate_x.append(gate.reshape(data_len, 4, 1) * data['x'].reshape(data_len, 1, -1))
                    gate_context_nids.append(gate * data['X_contextnids'].reshape(data_len, 1))
                    gate_pragmanids.append(gate * data['X_pragmanids'].reshape(data_len, 1))
                    gate_pragmascopenids.append(gate * data['X_pragmascopenids'].reshape(data_len, 1))
                    gate_pseudonids.append(gate * data['X_pseudonids'].reshape(data_len, 1))
                    gate_icmpnids.append(gate * data['X_icmpnids'].reshape(data_len, 1))
                
                else:
                    pseudonids = data['X_pseudonids'].bool()
                    # gate_x.append(gate.reshape(data_len, 4, 1) * data['x'][pseudonids].reshape(data_len, 1, -1))
                    gate_context_nids.append(gate * data['X_contextnids'][pseudonids].reshape(data_len, 1))
                    gate_pragmanids.append(gate * data['X_pragmanids'][pseudonids].reshape(data_len, 1))
                    gate_pragmascopenids.append(gate * data['X_pragmascopenids'][pseudonids].reshape(data_len, 1))
                    gate_pseudonids.append(gate * data['X_pseudonids'][pseudonids].reshape(data_len, 1))
                    gate_icmpnids.append(gate * data['X_icmpnids'][pseudonids].reshape(data_len, 1))
                    assert data['X_pragma_per_node'][~pseudonids].sum() == 0
                    pragma = data['X_pragma_per_node'][pseudonids]
                    gate_tile.append(gate * pragma[:, 0].bool().reshape(data_len, 1))
                    gate_pipeline.append(gate * (pragma[:, 1] != 0).reshape(data_len, 1))
                    gate_pipeline1.append(gate * (pragma[:, 1] == 1).reshape(data_len, 1))
                    gate_pipeline5.append(gate * (pragma[:, 1] == 5).reshape(data_len, 1))
                    gate_pipeline10.append(gate * (pragma[:, 1] == 10).reshape(data_len, 1))
                    gate_parallel1.append(gate * ((pragma[:, 3] > 0) & (pragma[:, 3] <= 4)).reshape(data_len, 1))
                    gate_parallel2.append(gate * (pragma[:, 3] > 4).reshape(data_len, 1))
                    gate_nopragmanids.append(gate * (pragma.sum(1) == 0).reshape(data_len, 1))
        
        gate_context_nids = torch.cat(gate_context_nids).mean(0)
        gate_pragmanids = torch.cat(gate_pragmanids).mean(0)
        gate_pragmascopenids = torch.cat(gate_pragmascopenids).mean(0)
        gate_pseudonids = torch.cat(gate_pseudonids).mean(0)
        gate_icmpnids = torch.cat(gate_icmpnids).mean(0)
        print('context nodes:', gate_context_nids / gate_context_nids.sum())
        print('pragma nodes:', gate_pragmanids / gate_pragmanids.sum())
        print('pragma scope nodes:', gate_pragmascopenids / gate_pragmascopenids.sum())
        print('pseudo nodes:', gate_pseudonids / gate_pseudonids.sum())
        print('icmp nodes:', gate_icmpnids / gate_icmpnids.sum())
        # gate_x = torch.cat(gate_x).mean(0)
        # gate_x = gate_x / gate_x.sum(0)
        # gate_x_std = gate_x.std(0)
        # rank_list = torch.argsort(gate_x_std)
        # cnt = 0
        # for i in range(1, len(rank_list) + 1):
        #     if gate_x[:, rank_list[-i]].isnan().any():
        #         continue
        #     print(f'{rank_list[-i]} {gate_x[:, rank_list[-i]]}')
        #     cnt += 1
        #     if cnt == 5:
        #         break
        
        if FLAGS.observe_moe_layer == 'pseudo_alone_w/o_gnn7':
            gate_nopragmanids = torch.cat(gate_nopragmanids).mean(0)
            gate_tile = torch.cat(gate_tile).mean(0)
            gate_pipeline = torch.cat(gate_pipeline).mean(0)
            gate_pipeline1 = torch.cat(gate_pipeline1).mean(0)
            gate_pipeline5 = torch.cat(gate_pipeline5).mean(0)
            gate_pipeline10 = torch.cat(gate_pipeline10).mean(0)
            gate_parallel1 = torch.cat(gate_parallel1).mean(0)
            gate_parallel2 = torch.cat(gate_parallel2).mean(0)
            print('no pragma nodes:', gate_nopragmanids / gate_nopragmanids.sum())
            print('gate tile:', gate_tile / gate_tile.sum())
            print('gate pipeline:', gate_pipeline / gate_pipeline.sum())
            print('gate pipeline 1:', gate_pipeline1 / gate_pipeline1.sum())
            print('gate pipeline 5:', gate_pipeline5 / gate_pipeline5.sum())
            print('gate pipeline 10:', gate_pipeline10 / gate_pipeline10.sum())  # This case is too few, not meaningful
            print('gate parallel 1:', gate_parallel1 / gate_parallel1.sum())
            print('gate parallel 2:', gate_parallel2 / gate_parallel2.sum())
    
    elif FLAGS.observe_moe_layer == 'output_mlp':
        with torch.no_grad():
            kernel_dict = {}
            gate_pragma_per_node = []
            for data in tqdm(train_loader):
                data = data.to(FLAGS.device)
                gate = model(data, return_gate=FLAGS.observe_moe_layer)
                kernel = data['gname'][0]
                try:
                    kernel_dict[kernel].append(gate)
                except:
                    kernel_dict[kernel] = [gate]
            for k, v in kernel_dict.items():
                v = torch.cat(v).mean(0)
                print(k, v)


# Save the hidden representations of the model
def save_hidden(train_loader, model):
    h6_sum_list, h6_mean_list, h7_list, h7_program_list, h7_pragma_list, perf_list = {}, {}, {}, {}, {}, {}
    with torch.no_grad():
        for data in tqdm(train_loader):
            kernel = data['kernel'][0].split('_')[0]
            data = data.to(FLAGS.device)
            hidden = model(data, return_middle=7)
            if kernel in h7_list.keys():
                h7_list[kernel].append(hidden.cpu().numpy())
                perf_list[kernel].append(data['perf'].item())
            else:
                h7_list[kernel] = [hidden.cpu().numpy()]
                perf_list[kernel] = [data['perf'].item()]
    
    for k in h7_list.keys():
        h7_list[k] = np.concatenate(h7_list[k], 0)
        np.save(f'hidden_save/{k}_h7.npy', h7_list[k])
        np.save(f'hidden_save/{k}_perf.npy', perf_list[k])
        print(f'saving at hidden_save/{k}_h7.npy and hidden_save/{k}_perf.npy')


def save_pred_for_contest(train_loader, model):
    model.eval()
    if FLAGS.task == 'class':
        out_dict = {'valid': [], 'key': [], 'class_score': []}
    else:
        out_dict = {'perf': [], 'util-LUT': [], 'util-DSP': [], 'util-FF': [], 'util-BRAM': [], 'key': [],
            'perf_label': [], 'actual_perf_label': [], 'util-LUT_label': [], 'util-DSP_label': [],
            'util-FF_label': [], 'util-BRAM_label': []}
    
    with torch.no_grad():
        for data in tqdm(train_loader):
            data = data.to(FLAGS.device)
            output = model(data, return_middle=8)
            keys = data['key']
            kernels = data['gname']
            for i in range(len(keys)):
                keys[i] = f"__version__-v21.__kernel__-{kernels[i]}.{keys[i]}"

            out_dict['key'].extend(keys)
            if FLAGS.task == 'class':
                pred = output['perf']
                pred_score = torch.softmax(pred, dim=1)
                pred_score = pred_score.max(1).values.cpu().numpy()
                pred = pred.cpu().numpy()
                pred = pred[:, 1] > pred[:, 0]
                out_dict['valid'].extend(pred)
                out_dict['class_score'].extend(pred_score)
            else:
                out_dict['perf'].extend(output['perf'][:, 0].cpu().numpy())
                out_dict['util-LUT'].extend(output['util-LUT'][:, 0].cpu().numpy())
                out_dict['util-DSP'].extend(output['util-DSP'][:, 0].cpu().numpy())
                out_dict['util-FF'].extend(output['util-FF'][:, 0].cpu().numpy())
                out_dict['util-BRAM'].extend(output['util-BRAM'][:, 0].cpu().numpy())
                out_dict['perf_label'].extend(data['perf'].cpu().numpy())
                out_dict['actual_perf_label'].extend(data['actual_perf'].cpu().numpy())
                out_dict['util-LUT_label'].extend(data['util_LUT'].cpu().numpy())
                out_dict['util-DSP_label'].extend(data['util_DSP'].cpu().numpy())
                out_dict['util-FF_label'].extend(data['util_FF'].cpu().numpy())
                out_dict['util-BRAM_label'].extend(data['util_BRAM'].cpu().numpy())
    
    df_new = pd.DataFrame(out_dict)
    file_name = f'prediction{FLAGS.save_pred_name}.csv'

    if os.path.exists(file_name):
        df_existing = pd.read_csv(file_name)
        if len(df_existing) != len(df_new):
            raise ValueError("The number of rows in the existing file does not match the dictionary.")
        assert set(df_existing['key']) == set(df_new['key'])
        columns1 = set(df_existing.columns) - {"key"}
        columns2 = set(df_new.columns) - {"key"}
        assert len(columns1.intersection(columns2)) == 0
        df_combined = pd.merge(df_existing, df_new, on='key')
    else:
        df_combined = df_new
    df_combined.to_csv(file_name, index=False)


def train_main(dataset, pragma_dim = None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.test_ratio, resample=-1):
    global saver
    saver.info(f'Reading dataset from {SAVE_DIR}')
    
    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    if FLAGS.finetune:
        assert dataset_dict['test'] == None
        if FLAGS.train_mode in ['save_hidden', 'save_moe', 'direct_test', 'save_pred_for_contest', 'observe_moe_distribution']:
            file_li = dataset.processed_file_names
            dataset = MyOwnDataset(data_files=file_li)
            li = [dataset, [], []]
        else:
            if FLAGS.transfer_k_shot > 1:
                _m_val = r2
            else:
                _m_val = val_ratio
            li, raw_li, kernel_lengths = split_dataset(dataset_dict['train'], FLAGS.transfer_k_shot, _m_val,
                dataset_test=dataset_dict['test'], pragma_dim=pragma_dim,
                num_features=153, edge_dim=335)
    elif resample == -1:
        li = split_dataset(dataset_dict['train'], r1, r2, dataset_test=dataset_dict['test'])
    else:
        li = split_dataset_resample(dataset_dict['train'], 1.0 - val_ratio - test_ratio, val_ratio, test_ratio, test_id=resample)
    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    
    if len(FLAGS.moe_layers) > 0 and FLAGS.moe_layers[0][0:9] == 'hierarchy':
        assert len(FLAGS.moe_layers) == 1
        model = HierarchicalMoE(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    else:
        model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    # print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print('Number of total parameters:', sum(p.numel() for p in model.parameters()))
    # exit()
    
    if FLAGS.model_path != None:
        model_path = FLAGS.model_path[0] if type(FLAGS.model_path) is list else FLAGS.model_path 
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
    model = model.to(FLAGS.device)
    
    if FLAGS.finetune:
        if FLAGS.train_mode == 'save_hidden':
            save_hidden(train_loader, model)
            exit()
        elif FLAGS.train_mode == 'save_moe':
            save_moe(train_loader, model)
            exit()
        elif FLAGS.train_mode == 'direct_test':
            test_loss, loss_dict_test, gae_loss_test, _ = test(train_loader, 'test', model, 0, False, [])
            print(f'initial test loss: {test_loss:.4f}')
            exit()
        elif FLAGS.train_mode == 'observe_moe_distribution':
            observe_moe_distribution(train_loader, model)
            exit()
        elif FLAGS.train_mode == 'save_pred_for_contest':
            save_pred_for_contest(train_loader, model)
            exit()

    saver.log_model_architecture(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=float(FLAGS.weight_decay))
    num_steps = len(train_loader) * FLAGS.epoch_num
    
    if FLAGS.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[FLAGS.epoch_num // 3], gamma=0.1)
    elif FLAGS.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    if FLAGS.warmup == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif FLAGS.warmup == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif FLAGS.warmup == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)

    # if len(test_loader) > 0:
        # test_loss, loss_dict_test, gae_loss_test, _ = test(test_loader, 'test', model, 0, plot_test, test_losses)
        # print(f'initial test loss: {test_loss:.4f}')
        # exit()
    
    train_losses, val_losses, test_losses, total_lrs = [], [], [], []
    gae_train_losses, gae_val_losses, gae_test_losses = [], [], []
    plot_test = False
    
    # TODO: use the test result at the best epoch
    best_val, final_test, best_test = 10000, 10000, 10000
    for epoch in range(FLAGS.epoch_num):
        plot_test = False
        timer = OurTimer()
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        saver.log_info(f'Test batch ID (resample): {resample} - Epoch {epoch} train')
        loss, loss_dict_train, gae_loss_train, lrs = train(epoch, model, train_loader, optimizer, lr_scheduler,
            warmup_scheduler)
        plot_test = model_update(model, train_losses, loss, epoch, plot_test, 'train', saver)
        total_lrs.extend(lrs)

        if epoch == 0:
            torch.cuda.empty_cache()
        
        if epoch % 10 == 9 or epoch == 0:
            if len(val_loader) > 0:
                saver.log_info(f'Epoch {epoch} val')
                val, loss_dict_val, gae_loss_val, _ = test(val_loader, 'val', model, epoch)
                plot_test = model_update(model, val_losses, val, epoch, plot_test, 'val', saver)
            if len(test_loader) > 0:
                saver.log_info(f'Epoch {epoch} test')
                test_loss, loss_dict_test, gae_loss_test, _ = test(test_loader, 'test', model, epoch, plot_test, test_losses)
                plot_test = model_update(model, test_losses, test_loss, epoch, plot_test, 'test', saver)
            if len(val_loader) > 0 and len(test_loader) > 0:
                if val < best_val:
                    best_val = val
                    final_test = test_loss
            elif len(test_loader) > 0:
                if loss < best_val:
                    best_val = loss
                    final_test = test_loss
            best_test = min(best_test, test_loss)
    
            log_loss(loss_dict_train, gae_loss_train, "Train", saver)
            if len(val_loader) > 0 and len(test_loader) > 0:
                log_loss(loss_dict_val, gae_loss_val, "Val", saver)
                log_loss(loss_dict_test, gae_loss_test, "Test", saver)
                saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, '
                            'Test: {:.4f}) Time: {}'.format(
                            epoch, loss, val, test_loss, timer.time_and_clear())))
                gae_val_losses.append(gae_loss_val)
                gae_test_losses.append(gae_loss_test)
            elif len(test_loader) > 0:
                log_loss(loss_dict_test, gae_loss_test, "Test", saver)
                saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                                'Test: {:.4f}) Time: {}'.format(
                                epoch, loss, test_loss, timer.time_and_clear())))
                gae_test_losses.append(gae_loss_test)
            else:
                saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                                'Time: {}'.format(
                    epoch, loss, timer.time_and_clear())))
            gae_train_losses.append(gae_loss_train)
            
            if len(train_losses) > 50 and len(test_loader) > 0:
                if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                    break
        
            torch.cuda.empty_cache()
    
    print(f'Best val: {best_val:.4f}, final test: {final_test:.4f}, best test: {best_test:.4f}')
    print(f'Saved model at {saver.model_logdir}')
    exit()

    epochs = range(epoch+1)
    plot_loss_trend(epochs, train_losses, val_losses, test_losses, saver.get_log_dir(), file_name='losses.png')
    if FLAGS.gae_T or FLAGS.gae_P:
        plot_loss_trend(epochs, gae_train_losses, gae_val_losses, gae_test_losses, saver.get_log_dir(), file_name='gae_losses.png')
    if len(test_loader) > 0:
        saver.log_info(f'min test loss at epoch: {test_losses.index(min(test_losses))}')
    if len(val_loader) > 0:
        saver.log_info(f'min val loss at epoch: {val_losses.index(min(val_losses))}')
    if FLAGS.scheduler is not None:
        plot_lr_trend(total_lrs, FLAGS.epoch_num + 1, saver.get_log_dir())
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')


def train_maml_main(dataset, pragma_dim = None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.test_ratio, resample=-1):
    global saver
    saver.info(f'Reading dataset from {SAVE_DIR}')
    num_features = dataset[0].num_features
    saver.info(f'num features for training: {num_features}')
    edge_dim = dataset[0].edge_attr.shape[1]
    if len(FLAGS.moe_layers) > 0 and FLAGS.moe_layers[0][0:9] == 'hierarchy':
        assert len(FLAGS.moe_layers) == 1
        model = HierarchicalMoE(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    else:
        model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    assert FLAGS.model_path == None and FLAGS.finetune == False

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
    model = model.to(FLAGS.device)

    kernel_dict = {}
    for file_name in dataset.processed_file_names:
        data = torch.load(file_name)
        kernel_name = data['gname']
        if kernel_name in kernel_dict.keys():
            kernel_dict[kernel_name].append(file_name)
        else:
            kernel_dict[kernel_name] = [file_name]

    saver.log_model_architecture(model)
    maml = l2l.algorithms.MAML(model, lr=FLAGS.lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), lr=FLAGS.lr, weight_decay=float(FLAGS.weight_decay))
    num_steps = FLAGS.epoch_num * FLAGS.MAML_num_kernel
    train_losses, val_losses = [], []
    
    # TODO: use the test result at the best epoch
    best_train, final_val, best_val = 10000, 10000, 10000
    for epoch in range(FLAGS.epoch_num):
        plot_test = False
        timer = OurTimer()
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        train_loss, val_loss = train_maml(epoch, maml, optimizer, kernel_dict)
        saver.log_info(f'Test batch ID (resample): {resample} - Epoch {epoch} train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')
        plot_test = model_update(model, train_losses, train_loss, epoch, plot_test, 'train', saver)
        plot_test = model_update(model, val_losses, val_loss, epoch, plot_test, 'val', saver)

        if epoch % 10 == 0:
            torch.cuda.empty_cache()
        if train_loss < best_train:
            best_train = train_loss
            final_val = val_loss
            best_val = min(best_val, val_loss)
            saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, Time: {}'.format(
                epoch, train_loss, val_loss, timer.time_and_clear())))
    
    print(f'Best train: {best_train:.4f}, final val: {final_val:.4f}, best val: {best_val:.4f}')
    # save model
    if FLAGS.save_model:
        saver.log_info((f'Saved model at epoch {epoch}'))
        save_epoch = (int(epoch / 250) + 1) * 250
        torch.save(model.state_dict(), join(saver.model_logdir, f"{save_epoch}_train_model_state_dict.pth"))
        print(join(saver.model_logdir, f"{save_epoch}_train_model_state_dict.pth"))
    exit()


def set_target_list():    
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task =='regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    
    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0
        
    return target_list, loss_dict

def update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct):
    if FLAGS.task == 'regression':
        total_loss += loss.item() # * data.num_graphs
        if not FLAGS.SSL:
            for t in target_list:
                loss_dict[t] += loss_dict_[t].item()
        return loss_dict, total_loss
    else:
        loss_, pred = torch.max(out_dict[FLAGS.target[0]], 1)
        labels = _get_y_with_target(data, FLAGS.target[0])
        correct += (pred == labels).sum().item()
        total_loss += labels.size(0)
        return pred, correct, total_loss


def train_maml(epoch, maml, optimizer, kernel_dict):
    sampled_index = torch.randint(0, len(kernel_dict.keys()), (FLAGS.MAML_num_kernel,))
    sampled_kernels = [list(kernel_dict.keys())[i] for i in sampled_index]
    total_train_loss, total_val_loss = 0, 0
    optimizer.zero_grad()

    for kernel in tqdm(sampled_kernels):
        kernel_size = len(kernel_dict[kernel])
        train_size = int(kernel_size * FLAGS.MAML_train_ratio)
        val_size = kernel_size - train_size
        li = random_split(kernel_dict[kernel], [train_size, val_size])
        train_dataset = MyOwnDataset(data_files=li[0])
        val_dataset = MyOwnDataset(data_files=li[1])
        train_loader = DataLoader(train_dataset, batch_size=len(li[0]), shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=len(li[1]), shuffle=False, pin_memory=False, num_workers=0)
        assert len(train_loader) == len(val_loader) == 1
        
        learner = maml.clone()
        for data in train_loader:
            data = data.to(FLAGS.device)
            out_dict, train_loss, loss_dict_, gae_loss = learner(data)
            learner.adapt(train_loss)
        
        for data in val_loader:
            data = data.to(FLAGS.device)
            out_dict, val_loss, loss_dict_, gae_loss = learner(data)
            val_loss.backward()
        total_train_loss += train_loss.item()
        total_val_loss += val_loss.item()
    
    total_train_loss /= FLAGS.MAML_num_kernel
    total_val_loss /= FLAGS.MAML_num_kernel
    for p in maml.parameters():
        p.grad.data.mul_(1.0 / FLAGS.MAML_num_kernel)
    optimizer.step()
    return total_train_loss, total_val_loss


def train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler):
    model.train()
    lrs = []
    total_loss, correct, i = 0, 0, 0
    target_list, loss_dict = set_target_list()
    
    for data in tqdm(train_loader):
        # pseudo_ids = torch.nonzero(data['X_pseudonids'], as_tuple=True)[0]
        # edge_index = data['edge_index']
        # group_ids = torch.zeros_like(data['X_pseudonids'])
        # pseudo_id_set = set(pseudo_ids.numpy())
        # for i, pseudo_id in enumerate(pseudo_ids):
        #     edge_mask = (edge_index == pseudo_id)
        #     edge_mask = edge_mask[0] | edge_mask[1]
        #     oppo_id = edge_index[:, edge_mask].sum(0) - pseudo_id
        #     oppo_id_not_pseudo = set(oppo_id.cpu().numpy()) - pseudo_id_set
        #     oppo_id_not_pseudo = torch.tensor(list(oppo_id_not_pseudo))
        #     assert group_ids[oppo_id_not_pseudo].sum() == 0
        #     group_ids[pseudo_id] = i + 1
        #     group_ids[oppo_id] = i + 1
        # assert (group_ids == 0).sum() == 0
        # exit()

        if FLAGS.scheduler is not None:
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if i == 0:
                saver.log_info(f"epoch = {epoch}, learning rate = {lr}")
        if FLAGS.load_data_to_device == False:
            data = data.to(FLAGS.device)
        out_dict, loss, loss_dict_, gae_loss = model(data, epoch=epoch)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if FLAGS.scheduler is not None:
            lr_scheduler.step(lr_scheduler.last_epoch+1)
        if FLAGS.warmup is not None:
            warmup_scheduler.dampen()
        
        total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct)
        if FLAGS.task == 'regression': loss_dict, total_loss = total_loss_dict
        else: pred, correct, total_loss = total_loss_dict
        
        saver.writer.add_scalar('loss/loss', loss, epoch * len(train_loader) + i)
        if 'GNLL' in FLAGS.loss:
            for target in target_list:
                out = out_dict[target]
                sigma = out[:, 1].reshape(-1, 1)
                mu = out[:, 0].reshape(-1, 1)
                saver.writer.add_scalar(f'train/sigma-{target}', torch.mean(torch.sqrt(torch.exp(sigma))), epoch * len(train_loader) + i)
                saver.writer.add_scalar(f'train/mu-{target}', torch.mean(mu), epoch * len(train_loader) + i)
        i += 1
    
    if FLAGS.scheduler is not None and epoch < 2:
        create_dir_if_not_exists(join(saver.get_log_dir(), 'lrs'))
        # plot_lr_trend(lrs, epoch, join(saver.get_log_dir(), 'lrs'))
    if FLAGS.task == 'regression':
        return total_loss / len(train_loader), {key: v / len(train_loader) for key, v in loss_dict.items()}, gae_loss, lrs
    else:
        return 1 - correct / total_loss, {key: v / len(train_loader) for key, v in loss_dict.items()}, gae_loss, lrs


def inference_loss_function(pred, true):
    return (pred - true) ** 2

def update_csv_dict(csv_dict, data, i, target_name, target_value, out_value):
    if csv_dict is not None:
        gname = _get_y_with_target(data, 'gname')[i]
        pragma = _get_y_with_target(data, 'pragmas')[i][0].item()
        pragma = '-'.join([str(j.item()) for j in _get_y_with_target(data, 'pragmas')[i]])
        if True or 'blocked' in gname:
            if f'{gname}-{pragma}' not in csv_dict:
                csv_dict[f'{gname}-{pragma}'] = {'gname': gname, 'pragma': pragma}
            csv_dict[f'{gname}-{pragma}'][f'acutal-{target_name}'] = target_value
            csv_dict[f'{gname}-{pragma}'][f'predicted-{target_name}'] = out_value
            l = csv_dict['header']
            if f'acutal-{target_name}' not in l:
                l.extend([f'acutal-{target_name}', f'predicted-{target_name}'])
                csv_dict['header'] = l


def test(loader, tvt, model, epoch, plot_test = False, test_losses = [-1], csv_dict = None,
        data_list = [], is_train_set=False, is_val_set=False, alpha=None, gauss_sigma=None, train_hiddens=None):
    model.eval()
    my_softplus = nn.Softplus()
    inference_loss, correct, total, count_data = 0, 0, 0, 1
    points_dict = OrderedDict()
    target_list, loss_dict = set_target_list()
    for target_name in target_list:
        points_dict[target_name] = {'true': [], 'pred': [], 'sigma_mu': [], 'sigma+mu': [], 'sigma':[], 'error': []}
    with torch.no_grad():
        for data in tqdm(loader):
            if FLAGS.load_data_to_device == False:
                data = data.to(FLAGS.device)
            out_dict, loss, loss_dict_, gae_loss = model(data, test_mode=True, epoch=epoch)
            total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total, correct)
            if FLAGS.task == 'regression': loss_dict, total = total_loss_dict
            else: pred, correct, total = total_loss_dict  

            if not FLAGS.SSL:
                for target_name in target_list:
                    if 'inf' in FLAGS.subtask:
                        saver.info(f'{target_name}')
                    if FLAGS.task == 'class': out = pred
                    elif FLAGS.encode_log and 'perf' in target_name: out = out_dict['perf'] 
                    else: out = out_dict[target_name]
                        
                    if 'GNLL' in FLAGS.loss:
                        if FLAGS.loss == 'myGNLL':
                            sigma = torch.sqrt(torch.exp(out[:, 1].reshape(-1, 1)))
                        else:
                            sigma = torch.sqrt(my_softplus(out[:, 1].reshape(-1, 1)))
                        out_ = out
                        out = out[:, 0].reshape(-1, 1)
                    for i in range(len(out)):
                        out_value = out[i].item()
                        target_value = _get_y_with_target(data, target_name)[i].item()
                        if FLAGS.encode_log and target_name == 'actual_perf':
                            out_value = 2**(out_value) * (1 / FLAGS.normalizer)
                        if 'inf' in FLAGS.subtask:
                            inference_loss += inference_loss_function(out_value, target_value)
                            count_data += 1
                            update_csv_dict(csv_dict, data, i, target_name, target_value, out_value)
                                        
                            if FLAGS.get_forgetting_examples:
                                diff_pred = abs(out_value - target_value)
                                add = True if ('perf' in target_name and diff_pred > 0.5) or ('util' in target_name and diff_pred > 0.3) else False
                                if add:
                                    data_list.append(data)
                                    pragma_configs = [p if type(p) is str else str(p.item()) for p in _get_y_with_target(data, 'pragmas')[i]]
                                    saver.info(f"data {i} {_get_y_with_target(data, 'gname')[i]} pramga {'-'.join(pragma_configs)} actual value: {target_value:.2f}, predicted value: {out_value:.2f}")
                            elif out_value != target_value: # and sigma[i].item() > 0.57:
                                saver.info(f"{target_name} data {i} {_get_y_with_target(data, 'gname')[i]} pramga {_get_y_with_target(data, 'pragmas')[i][0].item()} actual value: {target_value:.2f}, predicted value: {out_value:.2f}") #, sigma: {sigma[i].item()}, log_var: {out_[i, 1].item()}')")
                            
                        points_dict[target_name]['pred'].append((target_value, out_value))
                        points_dict[target_name]['true'].append((target_value, target_value))
                        points_dict[target_name]['error'].append((target_value, abs(target_value - out_value)))
                        
                        if 'GNLL' in FLAGS.loss: # and FLAGS.subtask != 'inference':
                            points_dict[target_name]['sigma_mu'].append((target_value, out[i].item() - sigma[i].item()))
                            points_dict[target_name]['sigma'].append((target_value, sigma[i].item()))
                            points_dict[target_name]['sigma+mu'].append((target_value, out[i].item() + sigma[i].item()))

    if FLAGS.task != 'class' and FLAGS.plot_pred_points and tvt == 'test' and (plot_test or (test_losses and (total / len(loader)) < min(test_losses))):
        from utils import plot_points_with_subplot, plot_points_with_subplot_sigma
        saver.log_info(f'@@@ plot_pred_points')
        assert(isinstance(FLAGS.target, list))
        use_sigma = True if 'GNLL' in FLAGS.loss else False
        label = f'epoch_{epoch+1}_{tvt}_train' if is_train_set else f'epoch_{epoch+1}_{tvt}_test'
        if is_val_set: label = f'epoch_{epoch+1}_{tvt}_val'
        if 'inf' in FLAGS.subtask or 'GNLL' not in FLAGS.loss:
            plot_points_with_subplot(points_dict, label, saver.plotdir, target_list, use_sigma=use_sigma)
        if 'GNLL' in FLAGS.loss:
            plot_points_with_subplot_sigma(points_dict, label, saver.plotdir, target_list, use_sigma=use_sigma)
            

    if FLAGS.task == 'regression':
        if 'inf' in FLAGS.subtask:
            _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
        return (total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()}, gae_loss, inference_loss / count_data * len(target_list))
    else:
        if 'inf' in FLAGS.subtask: report_class_loss(points_dict)
        return 1 - correct / total, {key: v / len(loader) for key, v in loss_dict.items()}, gae_loss, 0
