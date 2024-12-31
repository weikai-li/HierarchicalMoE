import redis
import pickle5 as pickle
from collections import Counter
from os.path import join, dirname, abspath
from glob import glob, iglob
from copy import deepcopy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

from utils import get_root_path, load_pickle, save_pickle

# DATASET = 'cnn1'
DATASET = 'machsuite'
if DATASET == 'cnn1':
    DIR = './save/cnn1'
    KERNEL = 'cnn'
    gexf_file = './new_db/case1/CnnKernel1.cpp.gexf'
    db_path = './databases/cnn_case2/'
elif DATASET == 'machsuite':
    DIR = './save/machsuite/gemm-ncubed'
    KERNEL = 'stencil'
    gexf_file = './machsuite/dot-files/gemm-ncubed_kernel.cpp.gexf'
    db_path = './machsuite/databases/**/*'
    # db_path = './poly/databases/**/*'
    reference_point = []  
    reference_point.append({'__PARA__L0': 8, '__PARA__L1': 1, '__PARA__L2': 1, '__PIPE__L0': 'flatten', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 4, '__PARA__L1': 1, '__PARA__L2': 1, '__PIPE__L0': 'flatten', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 2, '__PARA__L1': 1, '__PARA__L2': 1, '__PIPE__L0': 'flatten', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 1, '__PARA__L2': 1, '__PIPE__L0': 'flatten', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 4, '__PARA__L2': 1, '__PIPE__L0': '', '__PIPE__L1': 'flatten', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 4, '__PARA__L2': 1, '__PIPE__L0': 'off', '__PIPE__L1': 'flatten', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 2, '__PARA__L2': 1, '__PIPE__L0': 'off', '__PIPE__L1': 'flatten', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 1, '__PARA__L2': 1, '__PIPE__L0': 'off', '__PIPE__L1': 'flatten', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 1, '__PARA__L2': 64, '__PIPE__L0': 'off', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 1, '__PARA__L2': 16, '__PIPE__L0': 'off', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})
    reference_point.append({'__PARA__L0': 1, '__PARA__L1': 1, '__PARA__L2': 8, '__PIPE__L0': 'off', '__PIPE__L1': 'off', '__TILE__L0': 1, '__TILE__L1': 1})

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
                        'nw', 'md', 'stencil-3d']
MACHSUITE_KERNEL = []
poly_KERNEL = ['atax', 'mvt', 'bicg', 'doitgen', 'gemm-p', 'gemver', '2mm', 
               'gesummv', '3mm', 'jacobi-1d', 'fdtd-2d', 'fdtd-2d-large',
               'adi', 'heat-3d', 'jacobi-2d', 'seidel-2d', 'doitgen-red',
               'trmm-opt', 'trmm', 'symm', 'covariance', 'gemm-p-large']  
# poly_KERNEL = ['adi', 'heat-3d', 'jacobi-2d', 'seidel-2d']
poly_KERNEL = [
    # 'doitgen-red',
    # 'trmm',
    #'trmm-opt',
    # 'atax-medium', 'bicg-medium', 'gesummv-medium', 'mvt-medium', 'syr2k',
    #'correlation'
    # 'atax', 'gesummv', 'mvt'
    'atax'
    ]


# run "redis-server" on command line first!
def get_db_object(db_path, KERNEL, VER, mode='min', normalize_perf=False, points_key_obj=None, key_points=None):
    # create a redis database
    database = redis.StrictRedis(host='localhost', port=6379)
    database.flushdb()
    
    # db_files = [f for f in glob(join(db_path, '*.db'))]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'merged' in f]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'merged' in f and 'updated3' in f]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'v20' not in f and 'single-merged' in f and 'yunsheng' not in f]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'v20' in f]p
    db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and f'{KERNEL}_' in f and 'large-size' not in f and VER in f and f'one-db-extended-round{round_num}' in f and 'archive' not in f and 'single-merged' not in f] # and 'round10' in f]
    print(db_files)
    if len(db_files) == 0:
        print(f'Warning: no database found for kernel {KERNEL} in ver {VER}')
        return
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    for idx, file in enumerate(db_files):
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        database.hmset(0, data)
        max_idx = idx + 1
    keys = [k.decode('utf-8') for k in database.hkeys(0)]

    keys = sorted(keys)
    points = {}
    zero_count = 0
    nonzero_count = 0
    diff_label = 0
    count_lv1_invalid = 0
    count_lv2_invalid = 0
    perf_seen = Counter()
    all_perfs = []
    count_filtered = 0
    count_filtered_nonzero = 0
    min_perf = float('inf')
    stop = True
    all_points = {}
    # print(f'{len(keys)} keys')
    for i in range(len(keys)):
        pickle_obj = database.hget(0, keys[i])
        obj = pickle.loads(pickle_obj.replace(b'localdse', b'autodse'))
        # if i < 1000:
        # if i == 0 or i == 85:
        if type(obj) is int or type(obj) is dict:
            continue
        # x = obj.ret_code
        if keys[i][0:3] == 'lv1' or obj.ret_code.name == 'TIMEOUT' or obj.perf < 100.0:
            continue
        if keys[i][0:3] == 'lv1' and obj.perf == 0:
            count_lv1_invalid += 1
        if keys[i][0:3] == 'lv2' and obj.perf == 0:
            count_lv2_invalid += 1
        # if '__PARA__L0' in obj.point:
        #     print(obj.point)
        
        if DATASET == 'cnn':    
            if keys[i][0:3] == 'lv2' and obj.point['__PIPE__L10'] == 'off' and obj.point['__TILE__L11'] == 1 and \
                obj.point['__TILE__L10'] == 1:
                count_filtered += 1
            if keys[i][0:3] == 'lv2' and obj.perf != 0.0 and obj.point['__PIPE__L10'] == 'off' and \
                obj.point['__TILE__L11'] == 1 and obj.point['__TILE__L10'] == 1:
                count_filtered_nonzero += 1
        elif not stop: 
            for reference in reference_point:
                found = True
                for p in reference:
                    if obj.point[p] != reference[p]:
                        found = False
                        #print(f'different at {p}, {obj.point[p]} vs {reference_point[p]}')
                        break
                if found:
                    print(f'point exists with perf: {obj.perf}')
                    attrs = vars(obj)
                    print(', '.join("%s: %s" % item for item in attrs.items()))
        # print(i)
        
        not_found = False
        count_non_default = 0
        if mode == 'default' or mode == 'only_one_pragma':
            for p, p_val in obj.point.items():
                # print(p, p_val, obj.point)
                if type(p_val) is str:
                    if p_val != 'off':
                        count_non_default += 1
                        # if mode == 'default':
                        #     not_found = True
                        #     break
                else:
                    if p_val != 1:
                        count_non_default += 1
                        # if mode == 'default':
                        #     not_found = True
                        #     break
            if mode == 'only_one_pragma':
                target = 1
            else:
                target = 0
            if count_non_default != target:
                not_found = True


            if not_found:
                continue
            else:
                # print(obj.point)
                # print(f'perf: {obj.perf} with util: {obj.res_util} and key: {keys[i]}')
                key_points[mode].append(keys[i])
                if mode == 'default': break
                else: continue
                    

        s = str(obj.point.values())
        all_points[(obj.perf, keys[i])] = (obj.point, obj.res_util, obj)
        if mode == 'min':
            points_key_obj[keys[i]] = (obj.perf, obj.res_util, obj)
        # if obj.perf == 4433.0:
        #     print('found')
        #     print(obj.res_util)
        # if obj.point['__PARA__L0'] == 64:
        #     print(s, obj.perf, keys[i])
        if obj.perf != 0 and obj.perf < min_perf:
            min_perf = obj.perf
            # print(min_perf, obj.point)
        if s not in points:
            points[s] = (i, keys[i], obj.perf, obj.ret_code)
            # points[s] = (i, keys[i], obj.perf, obj.ret_code.name)
            # print(keys[i])
            # print(obj.ret_code.name, 'perf', obj.perf)
            # print(len(keys[i].split('.')))
        else:
            # print(i, 'same as point', points[s][0])
            # print('@@@ new', keys[i], obj.perf, obj.ret_code.name)
            # print('### old', points[s][1], points[s][2], points[s][3])
            # print('keys the same?', keys[i][3:] == points[s][1][3:])
            diff_label_bool = obj.perf != points[s][2]
            # print('obj.perf different?', diff_label_bool)
            if diff_label_bool:
                diff_label += 1
            # if 'lv2' in keys[i] and obj.perf != 0:
            #     exit(-1)
        perf_seen[obj.perf] += 1
        if obj.perf != 0.0:
            all_perfs.append(obj.perf)
        if obj.perf == 0:
            zero_count += 1
        else:
            nonzero_count += 1

    # print(min(all_perfs))
    if len(all_points) > 0:
        f_db.close()
        # print(len(set(points)), len(points))
        # print('zero_count', zero_count, 'nonzero_count', nonzero_count,
        #       zero_count + nonzero_count, 'count_lv1_invalid', count_lv1_invalid,
        #       'count_lv2_invalid', count_lv2_invalid, 'filtered:', count_filtered,
        #       'count_filtered_nonzero', count_filtered_nonzero)
        # print('perf_seen', perf_seen)

        i = 0
        best_results_dict = {}
        print('############## TOP DESIGNS ##############')
        print(len(all_points))
        for (perf, key), (point, util, result) in sorted(all_points.items()):
            # print(key)
            # print(f'perf: {perf} for point: {point} with util: {util}')
            utils = {k[5:]: max(0.0, u) for k, u in util.items() if k.startswith('util-')}
            valid = all([utils[res] < 0.80001 for res in utils])
            # valid = True
            if valid:
                if util['util-LUT'] < 0.5:
                    print(f'perf: {perf} with util: {util} and key: {key}')
                    i += 1
                key_points[mode].append(key)
                best_results_dict[(perf, key)] = result
                if i >= 20:
                    break

    # with open(join(dirname(abspath(__file__)), 'timeouts', f'{KERNEL}.pickle'), 'wb') as handle:
    #     pickle.dump(best_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     handle.flush()

    if normalize_perf:
        new_perf_seen = Counter()
        new_all_perfs = []
        max_perf = max(perf_seen.keys())
        for k, v in perf_seen.items():
            reduction_rate = max_perf/k if k != 0 else 0
            new_perf_seen[reduction_rate] = v
            new_all_perfs.append(reduction_rate)
        print('normalized perf_seen', new_perf_seen)
        print()
        return new_perf_seen, new_all_perfs
    else:
        print()
        return perf_seen, all_perfs
    # return obj


def plot_scatter(points, label, save_dir='db_analysis/ff', connected = True):
    i = 0
    num_figure, x_dim, y_dim = 1, 1, 1
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    # fig.set_figheight(18)
    # fig.set_figwidth(24)
    m = {'all': 'o', 'min': 'x', 'default': '^', 'only_one_pragma': '*'}
    target_list = points.keys()
    for idx, target in enumerate(target_list):
        x_li = [p for p in points[target]]
        y_li = [val for _, val in points[target].items()]
        s = 2 if target == 'all' else 50
        plt.scatter(x_li, np.array(y_li), label=f'{target}', color=cs[idx % len(cs)], s=s, marker=m[target])   
        if target != 'all':
            for i, y in enumerate(y_li):
                ind = 1000 if i % 2 == 0 else -1000
                if 'perf' not in save_dir: ind = 0
                plt.annotate(y, (x_li[i], y+ind) , xytext=(-10,10), fontsize=6, 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.1)) #,
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
            #                 color='red'))
        plt.legend(loc='best')
        plt.title(f'ff')
        plt.axis('on')
        plt.xticks([])
        if 'perf' in save_dir:
            plt.yscale('log')
        points_dict = {}   
    
    plt.suptitle(f'{label}')    
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def plot_scatter_subplot(points, label, save_dir='db_analysis/all', connected = True):
    i = 0
    num_figure, x_dim, y_dim = 5, 3, 2
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'all': 'o', 'min': 'x', 'default': '^', 'only_one_pragma': '*'}
    target_list = points['perf'].keys()
    for id_, obj in enumerate(['perf', 'util-BRAM', 'util-DSP', 'util-FF', 'util-LUT']):
        plt.subplot(y_dim, x_dim, id_+1)
        cur_points = points[obj]
        for idx, target in enumerate(target_list):
            x_li = [p for p in cur_points[target]]
            y_li = [val for _, val in cur_points[target].items()]
            s = 2 if target == 'all' else 50
            plt.scatter(x_li, np.array(y_li), label=f'{target}', color=cs[idx % len(cs)], s=s, marker=m[target])   
            if target != 'all':
                for i, y in enumerate(y_li):
                    ind = 1000 if i % 2 == 0 else -1000
                    if 'perf' not in obj: ind = 0
                    plt.annotate(y, (x_li[i], y+ind) , xytext=(-10,10), fontsize=6, 
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.1)) #,
                # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                #                 color='red'))
        plt.legend(loc='best')
        plt.title(f'{obj}')
        plt.axis('on')
        plt.xticks([])
        if 'perf' in save_dir or 'perf' in obj:
            plt.yscale('log')
        points_dict = {}   
    
    plt.suptitle(f'{label}')    
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    
def log_dict_of_dicts_to_csv(fn, csv_dict, csv_header, delimiter=','):
    import csv
    fp = open(join('db_analysis', f'{fn}.csv'), 'w+')
    f_writer = csv.DictWriter(fp, fieldnames=csv_header)
    f_writer.writeheader()
    for d, value in csv_dict.items():
        if d == 'header':
            continue
        f_writer.writerow(value)
    fp.close()

if __name__ == '__main__':
    # VER = 'v18'
    # VER = 'v20'
    store = True
    if store:
        all_kernels_data = {}
        all_kernels_key_points = {}
        for VER in ['v20' , 'v18']:
            print(f'############ now processing tool {VER} ##########')
            round_num = 3 if VER == 'v20' else 13
            for dataset in ['machsuite', 'poly']: 
                if dataset == 'machsuite':
                    db_path = f'{get_root_path()}/machsuite/databases/**/*'
                    KERNELS = MACHSUITE_KERNEL
                elif dataset == 'poly':
                    db_path = f'{get_root_path()}/poly/databases/**/*'
                    KERNELS = poly_KERNEL
                for kernel in KERNELS:
                    print(f'now processing {kernel}')
                    points_key_obj, key_points = {}, {'min': [], 'default': [], 'only_one_pragma': []}
                    all_kernels_data[f'{VER}-{kernel}'] = points_key_obj
                    all_kernels_key_points[f'{VER}-{kernel}'] = key_points
                    for mode in ['min']: #, 'default', 'only_one_pragma']:
                        get_db_object(db_path, kernel, VER, mode=mode, points_key_obj=points_key_obj, key_points=key_points)

        save_pickle(all_kernels_data, f'all_kernels_data.pickle')
        save_pickle(all_kernels_key_points, f'all_kernels_key_points.pickle')

    else:
        all_kernels_data = load_pickle('all_kernels_data.pickle')
        all_kernels_key_points = load_pickle('all_kernels_key_points.pickle')
        plot_trend = True
        if plot_trend:
            for kernel, keys in all_kernels_data.items():
                points = {'all': {}, 'min': {}, 'default': {}, 'only_one_pragma': {}}
                points_all = {o: deepcopy(points) for o in ['perf', 'util-BRAM', 'util-DSP', 'util-FF', 'util-LUT']}
                id_, obj = 1, 'util-FF'
                for obj in ['util-BRAM', 'util-DSP', 'util-FF', 'util-LUT']:
                    points_all[obj]['all'] = {k: u[id_][obj] for k, u in sorted(keys.items(), key=lambda item: item[1][0])}
                    for mode in ['min', 'default', 'only_one_pragma']:
                        points_all[obj][mode] = {k: all_kernels_data[kernel][k][id_][obj] for k in all_kernels_key_points[kernel][mode]}
                id_, obj = 0, 'perf'
                points_all[obj]['all'] = {k: u[id_] for k, u in sorted(keys.items(), key=lambda item: item[1][0])}
                for mode in ['min', 'default', 'only_one_pragma']:
                    points_all[obj][mode] = {k: all_kernels_data[kernel][k][id_] for k in all_kernels_key_points[kernel][mode]}
                # print(points['min'])
                plot_scatter_subplot(points_all, kernel)
        
        count = {'min': 0, 'default': 0, 'only_one_pragma': 0}
        utils = {'util-BRAM': 0.0, 'util-DSP': 0, 'util-LUT': 0, 'util-FF': 0}
        utils_all = {'min': deepcopy(utils), 'default': deepcopy(utils), 'only_one_pragma': deepcopy(utils)}
        utils_all_max = deepcopy(utils_all)
        csv_dict = {'header' : ['kernel', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']}
        for kernel, keys in all_kernels_key_points.items():
            # count = {'min': 0, 'default': 0, 'only_one_pragma': 0}
            # utils_all = {'min': deepcopy(utils), 'default': deepcopy(utils), 'only_one_pragma': deepcopy(utils)}
            # utils_all_max = deepcopy(utils_all)
            print(f'now processing {kernel}')
            csv_dict[kernel] = {u: '' for u in utils}
            csv_dict[kernel]['kernel'] = kernel
            for mode in ['min', 'default', 'only_one_pragma']:
                # print('##################')
                # print(f'reporting mode {mode}')
                cur_keys = keys[mode]
                for k in cur_keys:
                    count[mode] += 1
                    obj = all_kernels_data[kernel][k]
                    for u in utils:
                        utils_all[mode][u] += obj[1][u]
                        utils_all_max[mode][u] = max(utils_all_max[mode][u], obj[1][u])
                    # print(f'perf: {obj[0]} with util: {obj[1]} and key: {k}')

        if count['min'] > 0:
            for mode in ['min', 'default', 'only_one_pragma']:
                print(f'# designs for {mode}: {count[mode]}')
                if count[mode] > 0:
                    print('##################')
                    print(f'reporting mode {mode}')
                    
                    print('avg')
                    for u in utils:
                        # csv_dict[kernel][u] += f'/{utils_all[mode][u] / count[mode]:.2f}'
                        print(f'{u}: {utils_all[mode][u] / count[mode]}', end=' ')
                    print('\nmax')
                    for u in utils:
                        print(f'{u}: {utils_all_max[mode][u]}', end=' ')
                    print()
                else:
                    pass
                    # for u in utils:
                    #     csv_dict[kernel][u] += f'/N'

        log_dict_of_dicts_to_csv(f'avg_util', csv_dict, csv_dict['header'])
    
    # # path = "/Users/yba/Documents/UCLA/Research/software/temp/all_dbs/svm_result.db"
    # # path = "./all_dbs/addweighted_result.db"
    # # path = "new_db/case1/result.db"
    # # path = 'databases/vitis/exhaustive/addweighted_result.db'
    # # path = 'databases/vitis/bottleneck/addweighted_result.db'

    # perf_seen = Counter()
    # all_perfs = []
    # from glob import glob
    # # files = glob('databases/vitis/bottleneck/*.db') + glob('databases/vitis/exhaustive/*.db')\
    # #         + glob('databases/cnn_case1/*.db') + glob('databases/cnn_case2/*.db')
    # files = glob('databases/vitis/exhaustive/*.db')\
    #     \
    #         + glob('databases/cnn_case1/*.db') + glob('databases/cnn_case2/*.db')

    # for i, path in enumerate(files):
    #     print(i, path)
    #     rtn1, rtn2 = get_db_object(path, normalize_perf=True)
    #     perf_seen.update(rtn1)
    #     all_perfs.extend(rtn2)
    # print(f'globally, {len(files)} files, invalid designs #: {perf_seen[0]}, '
    #       f'total designs #: {sum(perf_seen.values())}\n perf_seen {perf_seen}')

    # # path = 'databases/vitis/exhaustive/svm_result.db'
    # # path = 'databases/vitis/bottleneck/svm_result.db'
    # from utils import plot_dist

    # plot_dist(all_perfs, 'all_perfs', 'databases/vitis/', analyze_dist=True, bins=None)




    # # while obj.perf == 0.0:
    # #     print('Got 0.0 for perf... Repeating query...')
    # #     obj = get_db_object(svm_result_path)

    # exit(0)

    """
    # create a redis database
    database = redis.StrictRedis(host = 'localhost', port = 6379)
    
    # open a sample database
    f_db = open("./all_dbs/hog_result.db", "rb")
    
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    data = pickle.load(f_db)
    database.hmset(0, data)
    keys = [k.decode('utf-8') for k in database.hkeys(0)]
    # get a value of a specific key (key i)
    i = 6
    pickle_obj = database.hget(0, keys[i])
    obj = pickle.loads(pickle_obj)
    
    print(obj.ret_code, obj.perf, obj.res_util)  ## --> return code for the current configuration. checkout autodse/result.py for all available options
    # other variables that exists:
    ## obj.perf  --> the cycle counts of the given configuration
    ## obj.res_util --> the resource utilization for each type of the resourse (BRAM, DSP, LUT, FF)
    ##		    the variables "util_*" show the percentage of utilization
    ##		    the variables "total_*" show the exact number of resources used
    ## obj.eval_time --> how long it took for the tool to synthesize the given configuration
    ## obj.point --> the same as key. shows the value of each pragma
    ## obj.quality --> quality of the given design point as a measure of finite difference as defined in the paper 
    ##		   (ratio of difference of cycle and utilization compared to a base design)
    """
