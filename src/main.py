from config import FLAGS
from train import train_main, inference, train_maml_main
from dse import ExhaustiveExplorer, GeneticExplorer, SAExplorer
from saver import saver
from utils import get_root_path, load, get_src_path, plot_models_per_graph

from os.path import join
from glob import iglob

import config
TARGETS = config.TARGETS
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
simple_KERNEL = config.simple_KERNEL

# from data import get_data_list, MyOwnDataset
from data import get_data_list, MyOwnDataset, contest_get_data_list
import data


if __name__ == '__main__':
    print(FLAGS.task, FLAGS.moe_layers, FLAGS.data_loading_path, FLAGS.model_path)
    if not FLAGS.force_regen:
        dataset = MyOwnDataset()
        pragma_dim = load(join(data.SAVE_DIR, 'pragma_dim'))
        print('read dataset')
    else:
        if FLAGS.sample_finetune:
            from sample_finetune import sample_data_list
            dataset = sample_data_list()
        else:
            if 'contest' in FLAGS.data_loading_path:
                contest_get_data_list()
            else:
                dataset, pragma_dim = get_data_list()

    # TODO: I deactivate the following two lines
    # if FLAGS.encoder_path is not None:
        # pragma_dim = load(join(dirname(FLAGS.encoder_path), 'v18_pragma_dim'))

    def set_model_path(graph_type):
        path = 'logs/auto-encoder/round3-model-comparison/models/**'
        
        model = [f for f in iglob(join(get_src_path(), path), recursive=True) if f.endswith('.pth') and graph_type in f]
        assert len(model) == 1
        FLAGS.model_path = model[0]
        
        encoder = [f for f in iglob(join(get_src_path(), path), recursive=True) if f.endswith('.klepto') and graph_type in f]
        assert len(encoder) == 1
        FLAGS.encoder_path = encoder[0]

        if graph_type == 'initial': FLAGS.num_features, FLAGS.edge_dim, FLAGS.separate_pseudo = 134, 7, False
        elif graph_type == 'extended': FLAGS.num_features, FLAGS.edge_dim, FLAGS.separate_pseudo = 136, 298, True
        elif graph_type == 'hierarchy': FLAGS.num_features, FLAGS.edge_dim, FLAGS.separate_pseudo = 136, 299, True
        
    def vis_main(dataset):
        visualize_embeddings(dataset)
        
    def inf_main(dataset):
        if type(FLAGS.model_path) is not list:
            inference(dataset)
        else:
            for ind, model_path in enumerate(FLAGS.model_path):
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio)
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio, is_val_set=True)
                inference(dataset, init_pragma_dict=pragma_dim, model_path=model_path, model_id=ind, test_ratio=FLAGS.val_ratio, is_train_set=True)
                if ind + 1 < len(FLAGS.model_path):
                    saver.new_sub_saver(subdir=f'run{ind+2}')
                    saver.log_info('\n\n')


    if FLAGS.subtask == 'vis+inf':
        inf_main(dataset)
        vis_main(dataset)
    elif FLAGS.subtask == 'visualization' or FLAGS.subtask == 'cal_distance':
        vis_main(dataset)
    elif FLAGS.subtask == 'inference':
        inf_main(dataset)
    elif FLAGS.subtask == 'dse':
        if FLAGS.dataset == 'programl':
            first_dse = True
            if FLAGS.plot_dse: graph_types = ['initial', 'extended', 'hierarchy']
            else: graph_types = [FLAGS.graph_type]

            for dataset in ['machsuite', 'poly']:
                path = join(get_root_path(), 'dse_database', dataset, 'config')
                path_graph = join(get_root_path(), 'dse_database', 'programl', dataset, 'processed')
                if dataset == 'machsuite':   
                    KERNELS = MACHSUITE_KERNEL
                elif dataset == 'poly':
                    KERNELS = poly_KERNEL
                else:
                    raise NotImplementedError()
                # kernel = "gemm-blocked"   
                
                point = {'__PARA__L0': 2, '__PIPE__L0': 'off', '__TILE__L0': 1, '__PARA__L4': 32, \
                    '__PARA__L1': 1, '__PIPE__L1': 'flatten', '__TILE__L1': 1, '__PARA__L5': 1, '__PARA__L2': 1, \
                    '__PARA__L3': 8, '__PIPE__L3': '', '__TILE__L3': 1, '__PARA__L6': 16}
                #for kernel in MACHSUITE_KERNEL:
                for kernel in KERNELS:
                    if not FLAGS.all_kernels and not FLAGS.target_kernel in kernel:
                        continue
                    plot_data = {}
                    for graph_type in graph_types:
                        saver.info('*****************************************************************')
                        saver.info(f'Now processing {graph_type}')
                        if FLAGS.plot_dse:
                            set_model_path(graph_type)
                        saver.info('#################################################################')
                        saver.info(f'Starting DSE for {kernel}')
                        saver.debug(f'Starting DSE for {kernel}')
                        FLAGS.target_kernel = kernel
                        if FLAGS.explorer == 'exhaustive':
                            explorer = ExhaustiveExplorer(path, kernel, path_graph, first_dse = first_dse, run_dse = True, point = point, pragma_dim = pragma_dim)
                            if FLAGS.plot_dse: plot_data[graph_type] = explorer.plot_data
                        elif FLAGS.explorer == 'genetic':
                            GeneticExplorer(path, kernel, path_graph, first_dse = first_dse, run_dse = True, point = point, pragma_dim = pragma_dim)
                        elif FLAGS.explorer == 'annealing':
                            SAExplorer(path, kernel, path_graph, first_dse = first_dse, run_dse = True, point = point, pragma_dim = pragma_dim)
                        else:
                            raise NotImplementedError()
                        saver.info('#################################################################')
                        saver.info(f'')
                        first_dse = False

                    # break
                    if FLAGS.plot_dse:
                        plot_models_per_graph(saver.plotdir, kernel, graph_types, plot_data, FLAGS.target)
        elif FLAGS.dataset == 'simple-programl':
            first_dse = True
            path = join(get_root_path(), 'dse_database', 'simple-program', 'config')
            path_graph = join(get_root_path(), 'dse_database', 'simple-program', 'programl', 'processed', 'initial')   
            PRECISION = ['int', 'long', 'float', 'double']
            SIZE = [64, 128, 256, 512, 1024, 2048]
            for name in simple_KERNEL: 
                for precision in PRECISION:
                    for size in SIZE:
                        if 'unrolled' in name and size <= 128: # and (size <= 512):
                            continue
                        config_path = join(path, name)
                        kernel = f'{name}-{precision}-{size}'
                        saver.info('#################################################################')
                        saver.info(f'Starting DSE for {kernel}')
                        saver.debug(f'Starting DSE for {kernel}')
                        FLAGS.target_kernel = kernel
                        if FLAGS.explorer == 'exhaustive':
                            explorer = ExhaustiveExplorer(config_path, kernel, path_graph, first_dse = first_dse, run_dse = True, point = None, pragma_dim = pragma_dim)
                        first_dse = False
        else:
            raise NotImplementedError()
    else:
        test_ratio, resample_list = FLAGS.test_ratio, [-1]
        if FLAGS.resample:
            test_ratio, resample_list = 0.25, range(4)
        for ind, r in enumerate(resample_list):
            # saver.info(f'Starting training with resample {r}')
            if FLAGS.train_mode in ['normal', 'save_hidden', 'save_moe',
                    'direct_test', 'observe_moe_distribution', 'save_pred_for_contest']:
                test_data = train_main(dataset, pragma_dim, test_ratio=test_ratio, resample=r)
            elif FLAGS.train_mode == 'maml':
                assert FLAGS.finetune == False
                train_maml_main(dataset, pragma_dim, test_ratio=test_ratio, resample=r)
            if ind + 1 < len(resample_list):
                saver.new_sub_saver(subdir=f'run{ind+2}')
                saver.log_info('\n\n')

    saver.close()
