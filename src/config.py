### round 2 logs: v20 results
### round 12 logs: v18 results

from utils import get_user, get_host, get_src_path, get_save_path, get_root_path
import argparse
import torch
from glob import iglob
from os.path import join

decoder_arch = []

parser = argparse.ArgumentParser()
# TASK = 'class'
TASK = 'regression'
# TASK = 'rl'
parser.add_argument('--task', type=str, default=TASK)

# TODO: change subtasks
# SUBTASK = 'dse'
# SUBTASK = 'visualization'
# SUBTASK = 'cal_distance'
# SUBTASK = 'vis+inf'
# SUBTASK = 'inference'
SUBTASK = 'train'
parser.add_argument('--subtask', default=SUBTASK)
parser.add_argument('--plot_dse', default=False)


#################### visualization ####################
parser.add_argument('--vis_per_kernel', default=False) ## only tsne visualization for now 


######################## data ########################
TARGETS = ['perf', 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']

# TODO: change kernels
# MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
#                     'nw', 'md', 'stencil-3d']   # 9-3 kernels (unavailable: aes, md, spmv-crs)
MACHSUITE_KERNEL = []

# TODO: remove unseen kernels (including those that change the size, e.g., large/medium, 
# but do not need to remove those that change the dimension)
# Original kernels
# poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen',
#                'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver',
#                'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm',
#                'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
#                'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium', 'gemver-medium']  # 33 kernels
# pretraining kernels
# poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen',
#                'doitgen-red', 'fdtd-2d', 'gemm-p-large', 'gemver', 'gesummv', 'heat-3d',
#                'jacobi-1d', 'mvt', 'seidel-2d', 'symm', 'symm-opt', 'syrk', 'trmm', 'mvt-medium', 'correlation',
#                'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium']  # 27 kernels
# test kernels
poly_KERNEL = ['fdtd-2d-large', 'gemver-medium', 'syr2k', 'gemm-p', 'jacobi-2d', 'trmm-opt']
# 5 complex kernels
# poly_KERNEL = ['3d-rendering', 'att-3mm', 'att-3mm-fuse', 'spam-filter', 'vmmv']

simple_KERNEL = ['reduction', 'relu', 'mat-vec-sub-add', 'vec-mul-add-dep',
                '3vec-mul-add-stencil', 'messy-stencil-1d', 'unrolled-vec-add',
                'dot', '3vec-mul-element', '3vec-add-element', 'vec-mul', 'max-vec',
                'vec-mul-add', '2vec-add', 'cond-vec-mul-add', 'non-zero-vec'] # '1d-conv


parser.add_argument('--force_regen', type=bool, default=False)

parser.add_argument('--min_allowed_latency', type=float, default=100.0) ## if latency is less than this, prune the point (used when synthesis is not valid)
EPSILON = 1e-3
parser.add_argument('--epsilon', default=EPSILON)
NORMALIZER = 1e7
parser.add_argument('--normalizer', default=NORMALIZER)
parser.add_argument('--util_normalizer', default=1)
# MAX_NUMBER = 3464510.00
MAX_NUMBER = 1e10
parser.add_argument('--max_number', default=MAX_NUMBER)

norm = 'speedup-log2' # 'const' 'log2' 'speedup' 'off' 'speedup-const' 'const-log2' 'none' 'speedup-log2'
parser.add_argument('--norm_method', default=norm)
parser.add_argument('--new_speedup', default=True) # new_speedup: same reference point across all, 
                                                    # old_speedup: base is the longest latency and different per kernel

parser.add_argument('--invalid', type = bool, default=False) # False: do not include invalid designs

parser.add_argument('--encode_log', type = bool, default=False)
v_db = 'v21' # 'v21': Vitis2021.1 database, 'v20': Vitis2020.2 database, 'v18': SDx18.3 database
parser.add_argument('--v_db', default=v_db) 
if v_db != 'v18':
    parser.add_argument('--only_common_db', default=False)
    parser.add_argument('--test_extra', default=False)
    parser.add_argument('--only_new_points', default=False)
round_num = 1 if v_db == 'v21' else 3 if v_db == 'v20' else 13
parser.add_argument('--round_num', default=round_num) ## round number of retraining after data augmentation with DSE
parser.add_argument('--get_forgetting_examples', default=False)

test_kernels = None
# test_kernels = ['jacobi-1d', '3mm', 'fdtd-2d', 'gemm-p', 'gemver']
# if v_db == 'v20':
#     test_kernels = ['symm', 'fdtd-2d-large', 'trmm-opt']
# else:
#     test_kernels = ['fdtd-2d', 'jacobi-2d', 'trmm-opt'] ## to be used to split the kernels between training and testing. this is the list of test kernels
parser.add_argument('--test_kernels', default=test_kernels)
parser.add_argument('--dse_kernel', type=str, help='only used in new_dse.py')
parser.add_argument('--uid', type=str, help='only used in new_dse.py')
parser.add_argument('--redis_port', type=int, help='only used in new_dse.py')
target_kernel = None
# target_kernel = 'gemm-blocked'
parser.add_argument('--target_kernel', default=target_kernel)
if target_kernel == None:
    all_kernels = True
else:
    all_kernels = False
parser.add_argument('--all_kernels', type = bool, default=all_kernels)
keys_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/src/logs/regression_train_whole-machsuite-poly_2022-04-14T17-22-34.633421-MAML-9-seed100-sp-cos'
parser.add_argument('--keys_path', default=keys_path)
parser.add_argument('--sample_finetune', type = bool, default=False)

# dataset = 'vitis-cnn'
# dataset = 'machsuite'
# dataset = 'programl-machsuite'
dataset = 'programl' # machsuite and poly in GNN-DSE
# dataset = 'simple-programl' # lorenzo's dataset
parser.add_argument('--dataset', default=dataset)

benchmark = ['machsuite', 'poly']
if dataset == 'simple-programl':
    benchmark = ['simple']
parser.add_argument('--benchmarks', default=benchmark)


# tag = 'only-vitis'
# tag = 'stencil'
# tag = 'gemm-ncubed'
# tag = 'whole-machsuite'
tag = 'whole-machsuite-poly'
if dataset == 'simple-programl':
    tag = 'simple'
parser.add_argument('--tag', default=tag)


##################### graph type #####################
dac_baseline = False
graph_type = '' # original DAC22 graph
# graph_type = 'extended-pseudo-block-base' ## check that the connected ones are not used
# graph_type = 'extended-pseudo-block-connected'
graph_type = 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--graph_type', default=graph_type)
pragma_as_MLP, type_parallel, type_merge = True, '2l', '2l'
gnn_layer_after_MLP = 1
pragma_MLP_hidden_channels, merge_MLP_hidden_channels = None, None

################## iccad models ##################
if dac_baseline:
    gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = False, True, False, False, 0, 154, 7
    model_ver = 'DAC22'
elif 'hierarchy' not in graph_type: ## separate_PT original graph
    gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, True, False, True, 0.1, 154, 7
    model_ver = 'original-PT'
else:
    if pragma_as_MLP:
        if gnn_layer_after_MLP == 1: model_ver = 'best_post-gnn-2l'
        
        if type_parallel == '2l': pragma_MLP_hidden_channels = '[in_D // 2]'
        elif type_parallel == '3l': pragma_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        
        if type_merge == '2l': merge_MLP_hidden_channels = '[in_D // 2]'
        elif type_merge == '3l': merge_MLP_hidden_channels = '[in_D // 2, in_D // 4]'
        else: raise NotImplementedError()
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = False, True, True, False, 0.1, 153, 335
    else:
        gae_T, P_use_all_nodes, separate_pseudo, separate_T, dropout, num_features, edge_dim = True, False, False, True, 0.1, 156, 335   
        model_ver = 'hierarchy-PT'
        

################ one-hot encoder #################
encoder_path = None
encode_edge_position = True
fixed_path = False
use_encoder = True 
if use_encoder:
    # includes = ['round12-40kernel', f'MLP-{pragma_as_MLP}', graph_type, 'encoders']
    if pragma_as_MLP: includes = ['MLP']
    elif 'hierarchy' not in graph_type: includes = ['original']
    else: includes = ['hierarchy']
    encoder_path_list = [f for f in iglob(join(get_root_path(), 'encoders', '**'), recursive=True) if f.endswith('.klepto') and all(k in f for k in includes) and 'class' not in f]
    assert len(encoder_path_list) == 1, print(encoder_path_list)
    encoder_path = encoder_path_list[0]
        

parser.add_argument('--encoder_path', default=encoder_path)


################ model architecture #################
## self-supervised learning
SSL = False
parser.add_argument('--SSL', default = SSL)

## edge attributes
parser.add_argument('--encode_edge', type=bool, default=True)
parser.add_argument('--encode_edge_position', type=bool, default=encode_edge_position)

num_layers = 6
parser.add_argument('--num_layers', type=int, default=num_layers)  ### prev num_layer: 6
## after uniform type for all parallel reductions: 136, without pseudo node: 134
## simple-programl: 103 sept22
parser.add_argument('--num_features', default=num_features) # before 4-23: 142, all-data-all-dac: 139) first week of june 2022: 143, last week of july 2022: 145
## 22 kernels, Nov 1st: 142
parser.add_argument('--edge_dim', default=edge_dim) ## 299/298/7 for hierarchy/extended/original round 3

parser.add_argument('--no_pragma', type=bool, default=False)

multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
if SUBTASK == 'class':
    multi_target = ['perf']
# TODO: change this
# multi_target = ['perf']
## DAC'22
# multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
parser.add_argument('--target', default=multi_target)
parser.add_argument('--MLP_common_lyr', default=0)

parser.add_argument('--no_graph', type = bool, default=False)
parser.add_argument('--only_pragma', type = bool, default=False)
# gnn_type = 'gcn'
# gnn_type = 'gat'
gnn_type = 'transformer'
parser.add_argument('--gnn_type', type=str, default=gnn_type)
parser.add_argument('--dropout', type=float, default=dropout)

# jkn_mode = 'lstm'
jkn_mode = 'max'
parser.add_argument('--jkn_mode', type=str, default=jkn_mode)
parser.add_argument('--jkn_enable', type=bool, default=True)
node_attention = True
parser.add_argument('--node_attention', type=bool, default=node_attention)
data_loading_path = ""

if node_attention:
    parser.add_argument('--node_attention_MLP', type=bool, default=False)

    separate_P = True
    parser.add_argument('--separate_P', type=bool, default=separate_P)
    separate_icmp = False
    parser.add_argument('--separate_icmp', type=bool, default=separate_icmp)
    # separate_T = True
    parser.add_argument('--separate_T', type=bool, default=separate_T)
    # separate_pseudo = False
    if 'raw_graph' in data_loading_path:
        separate_pseudo = False
    parser.add_argument('--separate_pseudo', type=bool, default=separate_pseudo)

    if separate_P:
        # P_use_all_nodes = False
        parser.add_argument('--P_use_all_nodes', type=bool, default=P_use_all_nodes)
    if separate_T:
        encoder = False
        if dataset == 'simple-programl' or target_kernel is not None or encoder_path:
            encoder = False
        parser.add_argument('--pragma_encoder', type=bool, default=encoder)
    parser.add_argument('--pragma_uniform_encoder', type=bool, default=True)
    
## graph auto encoder
# gae_T = True
parser.add_argument('--gae_T', default = gae_T)
gae_P = False
parser.add_argument('--gae_P', default = gae_P)
if gae_P:
    parser.add_argument('--input_encode', default = False)
    # d_type = 'None'
    # d_type = 'type1'
    d_type = 'type1'
    parser.add_argument('--decoder_type', default = d_type)

if pragma_as_MLP:
    assert graph_type == 'extended-pseudo-block-connected-hierarchy'
parser.add_argument('--gnn_layer_after_MLP', default=gnn_layer_after_MLP) ## number of message passing layers after MLP (pragma as MLP)
parser.add_argument('--pragma_as_MLP', default=pragma_as_MLP)
pragma_as_MLP_list = ['tile', 'pipeline', 'parallel']
parser.add_argument('--pragma_as_MLP_list', default=pragma_as_MLP_list)
pragma_scope = 'block'
parser.add_argument('--pragma_scope', default=pragma_scope)
keep_pragma_attribute = False if pragma_as_MLP else True
parser.add_argument('--keep_pragma_attribute', default=keep_pragma_attribute)
pragma_order = 'sequential'
pragma_order = 'parallel_and_merge'
parser.add_argument('--pragma_order', default=pragma_order)
# pragma_MLP_hidden_channels = None
# pragma_MLP_hidden_channels = '[in_D // 2]'
parser.add_argument('--pragma_MLP_hidden_channels', default=pragma_MLP_hidden_channels)
# merge_MLP_hidden_channels = '[in_D // 2]'
parser.add_argument('--merge_MLP_hidden_channels', default=merge_MLP_hidden_channels)


transfer_learning = False
if transfer_learning == False:
    model_path = None
else:
    model_path = ''
model_path_list = []


# TODO: change - use_pretrain if in inference, else false.
use_pretrain = False # False if (not FT_extra or SUBTASK != 'dse') else True
if use_pretrain:
    if 'hierarchy' in graph_type:#
        base_path = 'logs/auto-encoder/hierarchy/**'
        base_path = 'logs/auto-encoder/iccad/**'
    elif 'connected' in graph_type:
        base_path = 'logs/auto-encoder/extended-graph-db/**'
    else:
        base_path = 'logs/auto-encoder/all-data-sepPT/**'    
        base_path = 'logs/auto-encoder/iccad/**'    
        # base_path = 'logs/auto-encoder/**'    
        # base_path = 'logs/dac/**'
    model_tag = 'val_model'
    # if not separate_T: keyword = 'only-P'
    # elif P_use_all_nodes: keyword = 'with-mutual'
    # else: keyword = 'no-mutual'
    keyword, exclude_keyword = 'MSE', 'DAC22'
    if dac_baseline: keyword, exclude_keyword = 'DAC22', 'PT'
    elif 'hierarchy' in graph_type: 
        keyword, exclude_keyword = 'MSE', 'RMSE'
    
    if graph_type == 'extended-pseudo-block-connected-hierarchy': 
        if pragma_as_MLP: graph = pragma_order
        else: graph = 'hierarchy'
    elif graph_type == '': graph = 'orig'
    else: raise RuntimeError()

    excludes = ['dse', 'inference', 'visualization', 'vis+inf', 'bad', 'higher-loss', 'epoch10', 'gradually', 'icmp-vector', 'no-util', 'u5', exclude_keyword]
    # TODO: temporary
    # excludes = []
    # TODO: change
    includes = [f'round12-40kernel', f'position-{encode_edge_position}', f'{num_layers}L', f'gae-T-{gae_T}-gae-P-{gae_P}', graph, keyword, 'scheduler', 'gnn-after'] # 'sepPT', f'{num_layers}L #, 'fine-tune'] #, 'epoch10-only-gemm'] ## higher loss on gemm has lower loss on round9 test set
    # includes = ["erm", "train", "single"]
    if pragma_as_MLP:
        if type_parallel == '3l': includes.append('3l-parallel')
        elif type_parallel != '2l': 
            excludes.append('2l-parallel')
            if pragma_order == 'parallel_and_merge': includes.append('dropout0')
        else: includes.append('2l-parallel')

    if SUBTASK == 'dse' or 'vis' in SUBTASK or'cal' in SUBTASK or 'inf' in SUBTASK:
        keyword =  v_db
        includes = [keyword, model_ver]
        excludes = ['class']

    model_tag = 'val_model'
    for i in [0, 1, 2, 3]:
        # if SUBTASK == 'dse':
        #     keyword = f'freeze{i}'
        # elif SUBTASK == 'inference':
        #     keyword = f'-{i}.'
        #     keyword = f''
        # else:
        #     keyword = ''
        model_base_path = join(get_src_path(), base_path)
        if SUBTASK == 'dse' or 'vis' in SUBTASK or 'cal' in SUBTASK or 'inf' in SUBTASK: model_base_path = join(get_root_path(), 'models/**')
        model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and model_tag in f and all(k not in f for k in excludes) and all(k in f for k in includes)]
        print(model)
        assert len(model) == 1
        model_path = model[0]
        if SUBTASK == 'dse': # or SUBTASK == 'inference':
            model_path_list.append(model_path)
        else:
            model_path_list.append(model_path)
            break

if model_path_list != []:
    model_path = model_path_list
parser.add_argument('--model_path', default=model_path) ## list of models when used in DSE, if more than 1, ensemble inference must be on

ensemble = 0
ensemble_weights = None
# if model_path is not None:
#     ensemble = len(model_path)
#     ensemble_weights = [0.8447304,  -0.05980476,  0.1999986,   0.02298108]
parser.add_argument('--ensemble', type=int, default=ensemble)
parser.add_argument('--ensemble_weights', default=ensemble_weights)
class_model_path = None
if SUBTASK == 'dse':
    keyword =  v_db
    includes = [keyword, model_ver, 'class']
    model = [f for f in iglob(model_base_path, recursive=True) if f.endswith('.pth') and model_tag in f and all(k in f for k in includes)]
    assert len(model) == 1
    class_model_path = model[0]
parser.add_argument('--class_model_path', default=class_model_path)



################ transfer learning #################
feature_extract = False
parser.add_argument('--feature_extract', default=feature_extract) # if set to true GNN encoder (or part of it) will be fixed and only MLP will be trained
if feature_extract:
    parser.add_argument('--random_MLP', default=False) # true: initialize MLP randomly
fix_gnn_layer = None ## if none, all layers will be fixed
fix_gnn_layer = 1 ## number of gnn layers to freeze, feature_extract should be set to True
parser.add_argument('--fix_gnn_layer', default=fix_gnn_layer) # if not set to none, feature_extract should be True
FT_extra = False
parser.add_argument('--FT_extra', default=FT_extra) ## fine-tune only on the new data points

# TODO: change these settings
parser.add_argument('--finetune', default=transfer_learning)   # Only works when subtask = 'train'
parser.add_argument('--train_mode', default='normal', choices=['normal',
    'save_hidden', 'save_moe', 'maml', 'direct_test',
    'save_pred_for_contest', 'observe_moe_distribution'])
parser.add_argument('--save_pred_name', type=str, default="-3mm-moe")
parser.add_argument('--observe_moe_layer', default='gnn7',
                    choices=['gnn7', 'pseudo_alone_w/o_gnn7', 'output_mlp', 'hierarchy-weighted-hidden'])
parser.add_argument('--load_data_to_device', default=True)
parser.add_argument('--coreset', default='kmeans', choices=['random', 'kmeans'])
parser.add_argument('--transfer_k_shot', default=50)

parser.add_argument('--data_loading_path', type=str, default=data_loading_path)
parser.add_argument('--use_for_nodes', default=False)

# MoE settings
parser.add_argument('--moe_num_experts', type=int, default=4)
parser.add_argument('--moe_k', type=int, default=4)
parser.add_argument('--moe_lmbda', type=float, default=5e-3)
parser.add_argument('--hierarchical_moe_lmbda', type=float, default=5e-3)
parser.add_argument('--moe_layers', type=str, nargs='+', default=['hierarchy-weighted-hidden'],
                    choices=['gnn2', 'gnn3', 'gnn4', 'gnn5', 'gnn6', 'gnn7',
                    'pragma_mlp', 'pragma_merge', 'output_mlp', 'pseudo_alone', 'pseudo',
                    'pooling', 'pseudo_alone_w/o_gnn7', 'pseudo_alone+pooling',
                    'hierarchy-top-input', 'hierarchy-weighted-input', 'hierarchy-top-hidden',
                    'hierarchy-weighted-hidden'])   # "pseudo"-related MoE must be the first of the list
parser.add_argument('--hierarchical_moe_component', type=str, default=None,
                    choices=[None, 'gnn7', 'pseudo_alone_w/o_gnn7', 'output_mlp'])
parser.add_argument('--hierarchical_share_layers', default=False)
parser.add_argument('--hierarchical_alternate_train', default=True,
                    help='whether to alternately train in separate and joint after hierarchical_moe_epoch')

################ MAML settings #################
parser.add_argument('--MAML_num_kernel', default=32, help='How many kernels to sample per epoch')
parser.add_argument('--MAML_train_ratio', default=0.8)

################ training details #################
resample = False
val_ratio = 0.1
if resample or FT_extra: val_ratio = 0.0
parser.add_argument('--resample', default=resample) ## when resample is turned on, it will divide the dataset in round-robin and train multiple times to have all the points in train/test set
parser.add_argument('--val_ratio', type=float, default=val_ratio) # ratio of database for validation set
parser.add_argument('--test_ratio', type=float, default=0.1)        # ratio of database for test set

parser.add_argument('--activation', default='elu')     

parser.add_argument('--save_model', type = bool, default=True)

parser.add_argument('--D', type=int, default=64)

parser.add_argument('--lr', type=float, default=5e-4) ## default=0.001
scheduler, warmup, weight_decay = None, None, 0
scheduler, warmup, weight_decay = 'cosine', 'linear', 0.0001
parser.add_argument('--weight_decay', default=weight_decay) ## default=0.0001, larger than 1e-4 didn't help original graph P+T
parser.add_argument("--scheduler", default=scheduler)
parser.add_argument("--warmup", default=warmup)

# TODO: do not fix random seed
parser.add_argument('--random_seed', type=int, default=1) ## default=100
# TODO: Change it
batch_size = 64
# if graph_type != '':
#     batch_size = 32
parser.add_argument('--batch_size', type=int, default=batch_size)

loss = 'MSE' # RMSE, MSE, GNLL: Gaussian negative log likelihood of pytorch (predicting var), myGNLL: my implementation (predicting log_var)
parser.add_argument('--loss', type=str, default=loss) 
## now testing with out dim: 2 and nn.MSEloss, both targets get close to target
beta = 0 if loss != 'myGNLL' else 0.5
parser.add_argument('--beta', default=beta)

if not transfer_learning:
    if TASK == 'regression':
        epoch_num = 1000
    else:
        epoch_num = 200
    hierarchical_moe_epoch = epoch_num // 2
else:
    if TASK == 'regression':
        epoch_num = 500
    else:
        epoch_num = 200
    hierarchical_moe_epoch = 0

parser.add_argument('--epoch_num', type=int, default=epoch_num)
parser.add_argument('--hierarchical_moe_epoch', type=int, default=hierarchical_moe_epoch,
                    help='start training the hierarchical gate from this epoch')

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)



################# DSE details ##################
explorer = 'exhaustive'
# explorer = 'annealing'
# explorer = 'genetic'
parser.add_argument('--explorer', default=explorer)
if explorer == 'annealing':
    parser.add_argument('--init_temp', default = 100)
    parser.add_argument('--hls_temp_run', default = 10)
parser.add_argument('--dist_parent', default=True)
parser.add_argument('--dist_child', default=True)

model_tag = 'test'
parser.add_argument('--model_tag', default=model_tag)

parser.add_argument('--prune_util', default=True) # only DSP and BRAM
parser.add_argument('--prune_class', default=True)
pids = ['__PARA__L3', '__PIPE__L2', '__PARA__L1', '__PIPE__L0', '__TILE__L2', '__TILE__L0', '__PARA__L2', '__PIPE__L0']
parser.add_argument('--ordered_pids', default=pids)

if TASK == 'rl':
    parser.add_argument('--num_envs', type=int, default=2)


parser.add_argument('--print_every_iter', type=int, default=100)

plot = True
if SSL: plot = False
parser.add_argument('--plot_pred_points', type=bool, default=plot)


"""
HLS options

"""
parser.add_argument('--select_top10', action='store_false')
"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()
