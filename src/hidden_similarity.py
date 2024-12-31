import numpy as np
import json, os, time, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
sys.path.append('..')
from torch_geometric.loader import DataLoader
import random
from tqdm import tqdm

from model import Net
from config import FLAGS
from data import MyOwnDataset
from dse import ExhaustiveExplorer
from utils import get_root_path, dirname
from new_dse import random_sample
import RL.dse_utils as dse_utils
from parameter import compile_design_space
from saver import saver

import warnings
warnings.filterwarnings('ignore')


ensemble_KERNEL = ['gemm-blocked', 'gemm-ncubed', 'spmv-ellpack', 'stencil',
                   'nw', 'stencil-3d', '2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large',
                   'covariance', 'doitgen', 'doitgen-red', 'fdtd-2d', 'gemm-p-large', 'gemver', 'gesummv',
                   'heat-3d', 'jacobi-1d', 'mvt', 'seidel-2d', 'symm', 'symm-opt', 'syrk', 'trmm', 
                   'mvt-medium', 'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium']
test_KERNEL = ['fdtd-2d-large', 'gemver-medium', 'syr2k', 'gemm-p', 'jacobi-2d', 'trmm-opt']   # 'correlation'
# ensemble_KERNEL = ['gemm-blocked', 'gemm-ncubed', 'spmv-ellpack', 'stencil', 'stencil-3d', 'nw',
#         'doitgen', 'doitgen-red', '2mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance',
#         'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gesummv', 'heat-3d', 'jacobi-2d',
#         'seidel-2d', 'symm', 'symm-opt', 'syrk', 'syr2k', 'trmm', 'mvt-medium', 'correlation',
#         'atax-medium', 'bicg-medium', 'symm-opt-medium', 'gesummv-medium', 'gemver-medium',
#         'jacobi-1d', 'fdtd-2d', 'trmm-opt', '3mm', 'gemver', 'mvt']
# test_KERNEL = ['3d-rendering', 'att-3mm', 'att-3mm-fuse', 'knn', 'spam-filter', 'vmmv']


def load_input(kernel):
    try:
        input_list = np.load(f'{kernel}_input.npy')
    except:
        dataset = MyOwnDataset(e_kernel=kernel)
        data_files = dataset.processed_file_names
        dataset = MyOwnDataset(data_files=data_files)
        loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4)
        input_list = []
        for input in loader:
            input_list.append(input.pragmas)
        np.save(f'{kernel}_input.npy', input_list)
    return input_list


# feature = 'input' or 'hidden'
def visualize_hidden(use_color=True, feature='hidden'):
    hiddens = []
    lengths = []
    for k in ensemble_KERNEL:
        if feature == 'input':
            input_list = load_input(k)
            random.shuffle(input_list)
            print(len(input_list), input_list[0].shape)
            new_hidden = input_list
            hiddens.append(new_hidden)
            lengths.append(len(new_hidden))
        else:
            new_hidden = np.load(f'hidden_save/{k}_h7.npy')
            # random.shuffle(new_hidden)
            # new_hidden = new_hidden[:300]
            hiddens.append(new_hidden)
            lengths.append(len(new_hidden))
    hiddens = np.concatenate(hiddens, 0)

    test_hiddens = []
    test_lengths = []
    for k in test_KERNEL:
        if feature == 'input':
            input_list = load_input(k)
            random.shuffle(input_list)
            new_hidden = input_list
            test_hiddens.append(new_hidden)
            test_lengths.append(len(new_hidden))
        else:
            new_hidden = np.load(f'hidden_save/{k}_h7.npy')
            # random.shuffle(new_hidden)
            # new_hidden = new_hidden[:300]
            test_hiddens.append(new_hidden)
            test_lengths.append(len(new_hidden))
    test_hiddens = np.concatenate(test_hiddens, 0)
    hiddens = np.concatenate([hiddens, test_hiddens], 0)
    vis_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto').fit_transform(hiddens)

    plt.figure(figsize=(15, 15))
    len_sum = 0
    for i, new_len in enumerate(lengths):
        if use_color == True:
            plt.scatter(vis_emb[len_sum:len_sum+new_len,0], vis_emb[len_sum:len_sum+new_len,1],
                        linewidth=1, alpha=0.5)
        else:
            plt.scatter(vis_emb[len_sum:len_sum+new_len,0], vis_emb[len_sum:len_sum+new_len,1],
                        linewidth=1, color='black', alpha=0.5)
        len_sum += new_len
    plt.savefig(f'hidden_save/visual_hidden_base.png', bbox_inches='tight')
    for i, new_len in enumerate(test_lengths):
        plt.scatter(vis_emb[len_sum:len_sum+new_len,0], vis_emb[len_sum:len_sum+new_len,1],
            label=f'{test_KERNEL[i]}', linewidth=2, marker='*', alpha=1)
        len_sum += new_len
    plt.legend(prop={'size': 20})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'hidden_save/visual_hidden.png', bbox_inches='tight')
    plt.savefig(f'hidden_save/visual_hidden.pdf', bbox_inches='tight')
    plt.close()


def cos_similar(v1, v2):
    try:
        num = float(np.dot(v1, v2))
    except:
        print(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0


def calculate_sim():
    ensemble_hiddens = []
    for k in ensemble_KERNEL:
        ensemble_hiddens.append(np.load(f'hidden_save/{k}_h7.npy').mean(0))
    mean_list, min_list, max_list = [], [], []
    sim_mean_list, sim_min_list, sim_max_list = [], [], []
    for t in test_KERNEL:
        test_hidden = np.load(f'hidden_save/{t}_h7.npy').mean(0)
        print(test_hidden.shape)
        dis_list = []
        sim_list = []
        for h in ensemble_hiddens:
            dis = np.linalg.norm(test_hidden - h)
            sim = cos_similar(h, test_hidden)
            dis_list.append(dis)
            sim_list.append(sim)
        print(f'{t} distance: {np.mean(dis_list):.3f}, {np.min(dis_list):.3f}, {np.max(dis_list):.3f}')
        print(f'{t} similarity: {np.mean(sim_list):.3f}, {np.min(sim_list):.3f}, {np.max(sim_list):.3f}')
        mean_list.append(np.mean(dis_list))
        min_list.append(np.min(dis_list))
        max_list.append(np.max(dis_list))
        sim_mean_list.append(np.mean(sim_list))
        sim_min_list.append(np.min(sim_list))
        sim_max_list.append(np.max(sim_list))
    print(f'{np.mean(mean_list):.3f}, {np.mean(min_list):.3f}, {np.mean(max_list):.3f}')
    print(f'{np.mean(sim_mean_list):.3f}, {np.mean(sim_min_list):.3f}, {np.mean(sim_max_list):.3f}')


def visualize_moe(moe_layer: str):
    if moe_layer == 'output_mlp':
        group_ids = np.load('hidden_save/kernel_index.npy', allow_pickle=True).item()
    save_name = f'hidden_save/moe_{moe_layer}'
    gate = np.load(f'{save_name}_gate.npy')
    hidden = np.load(f'{save_name}_hidden.npy')
    sample_ratio = int(len(hidden) / 100000)
    if len(hidden) > 100000 and moe_layer != 'output_mlp':
        hidden = hidden[::sample_ratio]
        gate = gate[::sample_ratio]
    vis_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto').fit_transform(hidden)

    # plot hidden representation with the expert assignment
    plt.figure(figsize=(8, 8))
    if moe_layer == 'output_mlp':
        kernel_title_dict = {'fdtd-2d-large': 'Fd', 'gemver-medium': 'Gemv', 'gemm-p': 'Gemm',
                         'syr2k': 'Sy', 'jacobi-2d': 'Ja', 'trmm-opt': 'Tr'}
        for key, value in group_ids.items():
            if key == 'correlation':
                continue
            plt.scatter(vis_emb[value, 0], vis_emb[value, 1], label=kernel_title_dict[key])
        plt.legend(prop={'size': 16})
    else:
        plt.scatter(vis_emb[:, 0], vis_emb[:, 1])
    title_dict = {'gnn7': 'Node MoE', 'pseudo_alone_w/o_gnn7': 'Block MoE', 'output_mlp': 'Graph MoE'}
    plt.title(title_dict[moe_layer], size=14)

    if moe_layer == 'output_mlp':
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'./hidden_save/moe_{moe_layer}_hidden.png', bbox_inches='tight')
        print(f'save: hidden_save/moe_{moe_layer}_hidden.png')
        plt.close()
        plt.figure(figsize=(8, 8))
        one_idx = torch.ones(len(hidden))
        one_idx[group_ids['correlation']] = 0
        vis_emb = vis_emb[one_idx.bool()]
        gate = gate[one_idx.bool()]
    
    gate_addition = np.ones((len(gate), 1)) * 0.35
    gate = np.concatenate([gate, gate_addition], 1)
    gate_idx = np.argmax(gate, 1)
    for i in range(gate.shape[1] - 1):
        fit_idx = (gate_idx == i)
        plt.scatter(vis_emb[fit_idx, 0], vis_emb[fit_idx, 1], label=f'Expert {i}')
    fit_idx = (gate_idx == gate.shape[1] - 1)
    plt.scatter(vis_emb[fit_idx, 0], vis_emb[fit_idx, 1], label=f'Balanced', color='grey')
    plt.legend(prop={'size': 14})
    plt.xticks([])
    plt.yticks([])
    plt.title(title_dict[moe_layer], size=18)
    plt.savefig(f'./hidden_save/moe_{moe_layer}_gate.png', bbox_inches='tight')
    print(f'save: hidden_save/moe_{moe_layer}_gate.png')
    plt.close()


# Observe the search space of exhaustive search and AutoDSE w.r.t. the whole design space
def plot_search_space(kernel, search_limit, random_limit, force_regen=False):
    print(kernel)
    autodse_hidden = np.load(f'hidden_save/{kernel}_h7.npy')
    model_path = f'{get_root_path()}/src/logs/auto-encoder/iccad/round1-40kernel/pragma_as_MLP/parallel_and_merge/class/post-gnn-3lp-3lm/dropout-0.1-MSE-scheduler-warmup-weight-decay-0-no-mutual-info-PB-edge-attr-True_position-True-6L-SSL-False-gae-T-False-gae-P-False_regression_train_2024-03-02T02-16-24.443342/run1/1000_val_model_state_dict.pth'
    model = Net(153, edge_dim=335).to(FLAGS.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    try:
        if force_regen == True:
            exit()
        exhaustive_hidden = np.load(f'hidden_save/{kernel}_exhaustive_h7.npy')
        print('loaded exhaustive search results')
    except:
        explorer = ExhaustiveExplorer('../dse_database/poly/config/', kernel, '../dse_database/programl/poly/processed', {})
        explored_len = 0
        tqdm_bar = tqdm(search_limit)
        exhaustive_hidden = []
        for batch in explorer.gen():
            if explored_len >= search_limit:
                break
            data_list = []
            for point in batch:
                data_list.append(explorer.apply_design_point(explorer.graph, point, mode='regression', model=None))
            test_loader = DataLoader(data_list, batch_size=FLAGS.batch_size)
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(FLAGS.device)
                    new_hidden = model(data, return_middle=7)
                    exhaustive_hidden.extend(new_hidden.cpu().numpy())
            explored_len += len(batch)
            tqdm_bar.update(len(batch))
        np.save(f'hidden_save/{kernel}_exhaustive_h7.npy', exhaustive_hidden)
    try:
        assert len(exhaustive_hidden) >= search_limit
    except:
        print(f"[Warning] We only have {len(exhaustive_hidden)} number of exhaustive results")

    try:
        random_hidden = np.load(f'hidden_save/{kernel}_random_h7.npy')
        print(f'loaded random results with length = {len(random_hidden)}')
    except:
        random_hidden = []
    
    if len(random_hidden) < random_limit or force_regen == True:
        random_hidden = list(random_hidden)
        random_limit = random_limit - len(random_hidden)
        print(f'Need to generate {random_limit} number of random points')
        config_path = dse_utils.get_config_path('poly', kernel)
        # config_path: ../dse_database/poly/config/..._ds_config.json
        ds_config = dse_utils.load_config(config_path, saver)
        ds, num_ds = compile_design_space(ds_config['design-space']['definition'], None, saver)
        graph_path = dse_utils.get_graph_path('poly')
        # graph_path: ../dse_database/programl/poly/processed/extended-pseudo-block-connected-hierarchy/...
        graph = dse_utils.get_graph(graph_path, kernel)
        tqdm_bar = tqdm(random_limit)
        for _ in range(random_limit // FLAGS.batch_size + 1):
            sampled_points = random_sample(ds, FLAGS.batch_size, show_tqdm=False)
            data_list = []
            for point in sampled_points:
                data_list.append(dse_utils.apply_design_point(graph, point, mode='regression'))
            test_loader = DataLoader(data_list, batch_size=FLAGS.batch_size)
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(FLAGS.device)
                    new_hidden = model(data, return_middle=7)
                    random_hidden.extend(new_hidden.cpu().numpy())
                    tqdm_bar.update(len(data))
        np.save(f'hidden_save/{kernel}_random_h7.npy', random_hidden)

    # Remain 1 point in every 10 points
    exhaustive_hidden = exhaustive_hidden[::10]
    # random_hidden = random_hidden
    all_hidden = np.concatenate([autodse_hidden, exhaustive_hidden, random_hidden], 0)
    vis_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto').fit_transform(all_hidden)
    plt.figure(figsize=(12, 12))
    plt.title(kernel, size=20)
    plt.scatter(vis_emb[len(autodse_hidden) + len(exhaustive_hidden):, 0],
                vis_emb[len(autodse_hidden) + len(exhaustive_hidden):, 1], label='Random', color='black')
    plt.legend()
    plt.savefig(f'./visual/visual_{kernel}_random.png', bbox_inches='tight')
    plt.scatter(vis_emb[len(autodse_hidden): len(autodse_hidden) + len(exhaustive_hidden), 0],
                vis_emb[len(autodse_hidden): len(autodse_hidden) + len(exhaustive_hidden), 1], label='Exhaustive', color='green')
    plt.legend()
    plt.savefig(f'./visual/visual_{kernel}_exhaustive.png', bbox_inches='tight')
    autodse_perf = np.load(f'hidden_save/{kernel}_perf.npy')
    plt.scatter(vis_emb[:len(autodse_hidden), 0],
                vis_emb[:len(autodse_hidden), 1], label='AutoDSE', c=autodse_perf, cmap='magma', linewidths=3)
    plt.colorbar()
    plt.legend()
    plt.savefig(f'./visual/visual_{kernel}_all.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.title(kernel, size=20)
    plt.scatter(vis_emb[len(autodse_hidden) + len(exhaustive_hidden):, 0],
                vis_emb[len(autodse_hidden) + len(exhaustive_hidden):, 1], label='Random', color='black')
    plt.scatter(vis_emb[:len(autodse_hidden), 0],
                vis_emb[:len(autodse_hidden), 1], label='AutoDSE', c=autodse_perf, cmap='magma', linewidths=3)
    plt.colorbar()
    plt.legend()
    plt.savefig(f'./visual/visual_{kernel}_autodse.png', bbox_inches='tight')
    print(kernel)


def design_space_size(kernel):
    config_path = dse_utils.get_config_path('poly', kernel)
    # config_path: ../dse_database/poly/config/..._ds_config.json
    ds_config = dse_utils.load_config(config_path, saver)
    ds, num_ds = compile_design_space(ds_config['design-space']['definition'], None, saver)
    print(kernel, 'design space size:', num_ds)


# Based on the graph MoE assignment scores (seed=3 hierarchy-weighted-hidden model), calculate the similarity between kernels
def calculate_kernel_sim():
    # Hierarchical weights:
    hier_weights = {
        'fdtd_2d_large': [0.4761224091053009, 0.4034370183944702, 0.12044057250022888],
        'gemver_medium': [0.4278508722782135, 0.36628392338752747, 0.20586517453193665],
        'syr2k': [0.2489212602376938, 0.09355293959379196, 0.6575257778167725],
        'gemm_p': [0.3491555452346802, 0.436310350894928, 0.21453407406806946],
        'jacobi_2d': [0.28772732615470886, 0.2471790611743927, 0.46509361267089844],
        'trmm_opt': [0.26125818490982056, 0.4550236165523529, 0.2837182879447937],
        'doitgen_red': [0.2897207736968994, 0.24265392124652863, 0.46762529015541077],
        'mvt': [0.31235358119010925, 0.3224708139896393, 0.36517563462257385],
        'symm_opt': [0.3891943097114563, 0.11527109891176224, 0.49553462862968445],
        'gemm_p_large': [0.38667646050453186, 0.407223641872406, 0.20609992742538452],
        'adi': [0.3807472884654999, 0.185032457113266, 0.43422022461891174],
        'gemm_ncubed': [0.35794302821159363, 0.3104720413684845, 0.33158499002456665],
        'mvt_medium': [0.3889932334423065, 0.46742865443229675, 0.14357814192771912],
        'fdtd_2d': [0.37894415855407715, 0.4156774580478668, 0.20537829399108887],
        'trmm': [0.3170758783817291, 0.12757985293865204, 0.55534428358078],
        'gemm_blocked': [0.44560131430625916, 0.2944446802139282, 0.2599540054798126],
        '2mm': [0.2876530587673187, 0.322428822517395, 0.38991811871528625],
        'jacobi_1d': [0.38472336530685425, 0.3649318218231201, 0.250344842672348],
        'syrk': [0.2707449197769165, 0.1149211972951889, 0.6143339276313782],
        'stencil': [0.2640678286552429, 0.4358065724372864, 0.3001255691051483],
        'gemver': [0.34232157468795776, 0.34177324175834656, 0.3159051835536957],
        '3mm': [0.33442267775535583, 0.19326914846897125, 0.4723081886768341],
        'covariance': [0.33638796210289, 0.25430917739868164, 0.40930286049842834],
        'symm': [0.32367950677871704, 0.32893115282058716, 0.3473893702030182],
        'symm_opt_medium': [0.5275284647941589, 0.22705289721488953, 0.24541865289211273],
        'heat_3d': [0.5041815042495728, 0.29605865478515625, 0.1997598111629486],
        'atax_medium': [0.6754173636436462, 0.1945779174566269, 0.13000477850437164],
        'atax': [0.4604371190071106, 0.2374691516160965, 0.3020937740802765],
        'gesummv': [0.4478430449962616, 0.4245188534259796, 0.12763819098472595],
        'bicg_large': [0.4667937755584717, 0.42293310165405273, 0.11027316749095917],
        'seidel_2d': [0.11931964010000229, 0.6177603602409363, 0.2629200518131256],
        'spmv_ellpack': [0.17864620685577393, 0.5920997262001038, 0.22925400733947754],
        'stencil_3d': [0.48447465896606445, 0.2715635895729065, 0.24396181106567383],
        'bicg_medium': [0.4676584303379059, 0.44654422998428345, 0.08579721301794052],
        'bicg': [0.4444490671157837, 0.3967638611793518, 0.1587870717048645],
        'gesummv_medium': [0.42596620321273804, 0.4529610574245453, 0.12107278406620026],
        'nw': [0.475760817527771, 0.3760567903518677, 0.14818240702152252],
        'doitgen': [0.3481672704219818, 0.2899632751941681, 0.3618694841861725],
        'md': [0.16346701979637146, 0.6476600170135498, 0.1888730227947235],
    }

    # Graph MoE weights:
    graph_weights = {
        'fdtd_2d_large': [0.5157, 0.1997, 0.2432, 0.0414],
        'gemver_medium': [4.5669e-01, 3.1329e-01, 2.3001e-01, 1.6164e-05],
        'syr2k': [0.0214, 0.0697, 0.1006, 0.8083],
        'gemm_p': [0.1199, 0.1998, 0.3342, 0.3462],
        'jacobi_2d': [0.1788, 0.3103, 0.2706, 0.2403],
        'trmm_opt': [0.1840, 0.2995, 0.3087, 0.2078],
        'doitgen_red': [0.2229, 0.2698, 0.2766, 0.2306],
        'mvt': [0.0590, 0.2225, 0.2220, 0.4966],
        'symm_opt': [0.1468, 0.1614, 0.1323, 0.5594],
        'gemm_p_large': [0.2305, 0.3265, 0.3863, 0.0567],
        'adi': [0.5823, 0.2157, 0.1719, 0.0301],
        'gemm_ncubed': [0.1981, 0.4829, 0.2259, 0.0931],
        'mvt_medium': [0.2756, 0.3594, 0.3578, 0.0072],
        'fdtd_2d': [0.3094, 0.1701, 0.3575, 0.1629],
        'trmm': [0.3553, 0.3586, 0.1263, 0.1598],
        'gemm_blocked': [0.1590, 0.2090, 0.2154, 0.4167],
        '2mm': [0.1829, 0.5343, 0.2516, 0.0312],
        'jacobi_1d': [0.2895, 0.1916, 0.1675, 0.3515],
        'syrk': [0.0317, 0.0841, 0.1230, 0.7612],
        'stencil': [0.0485, 0.1086, 0.2127, 0.6302],
        'gemver': [0.2123, 0.3487, 0.1612, 0.2778],
        '3mm': [0.2388, 0.5022, 0.1736, 0.0854],
        'covariance': [0.1666, 0.2393, 0.2359, 0.3581],
        'symm': [0.1654, 0.2695, 0.3507, 0.2144],
        'symm_opt_medium': [0.3114, 0.3201, 0.2046, 0.1638],
        'heat_3d': [0.2716, 0.3649, 0.3471, 0.0164],
        'atax_medium': [6.4045e-01, 1.7803e-01, 1.8142e-01, 1.0606e-04],
        'atax': [0.3770, 0.1744, 0.1276, 0.3210],
        'gesummv': [0.2684, 0.3708, 0.3121, 0.0487],
        'bicg_large': [6.1411e-01, 1.5819e-01, 2.2764e-01, 6.4640e-05],
        'seidel_2d': [0.1806, 0.4577, 0.3132, 0.0485],
        'spmv_ellpack': [0.5649, 0.1663, 0.1724, 0.0964],
        'stencil_3d': [0.4370, 0.2810, 0.2030, 0.0789],
        'bicg_medium': [7.2038e-01, 1.2972e-01, 1.4950e-01, 4.0564e-04],
        'bicg': [0.3204, 0.1683, 0.2303, 0.2811],
        'gesummv_medium': [0.4746, 0.2876, 0.2313, 0.0065],
        'nw': [0.2264, 0.1832, 0.1949, 0.3955],
        'doitgen': [0.1063, 0.2151, 0.4216, 0.2571],
        'md': [0.6715, 0.1889, 0.1118, 0.0278]
    }

    # Combine these two statistics
    for key in hier_weights.keys():
        graph_weight = graph_weights[key]
        # graph_weight = [w * hier_weights[key][2] for w in graph_weight]
        # hier_weights[key].extend(graph_weight)
        hier_weights[key] = graph_weight

    test_kernels = ['fdtd_2d_large', 'gemver_medium', 'syr2k', 'gemm_p', 'jacobi_2d', 'trmm_opt']
    train_kernels = set(hier_weights.keys()) - set(test_kernels)
    
    for test_kernel in test_kernels:
        max_sim = 0
        best_kernels, best_is = [], []
        for i, train_kernel in enumerate(train_kernels):
            sim = cos_similar(hier_weights[test_kernel], hier_weights[train_kernel])
            if sim > max_sim - 1e-5:
                if abs(sim - max_sim) > 1e-5:
                    max_sim = sim
                    best_kernels = [train_kernel]
                    best_ids = [i]
                else:
                    max_sim = sim
                    best_kernels.append(train_kernel)
                    best_is.append(i)
        print(f'{test_kernel} is most similar to {best_kernels}, the similarity is {max_sim}')
        print(hier_weights[test_kernel])


if __name__ == '__main__':
    # calculate_sim()
    # visualize_hidden()
    # visualize_moe('output_mlp')
    # plot_search_space(kernel='att-3mm', search_limit=75000, random_limit=50000, force_regen=False)
    # design_space_size('syr2k')
    calculate_kernel_sim()
