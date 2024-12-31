import redis
import pickle
from collections import Counter
from os.path import join, dirname, abspath
from glob import glob, iglob

from utils import get_root_path

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
                        'nw']
poly_KERNEL = ['atax', 'mvt', 'bicg', 'doitgen', 'gemm-p', 'gemver', '2mm', 'gesummv', '3mm', 'jacobi-1d', 'fdtd-2d']  

# run "redis-server" on command line first!
def get_db_object(db_path, KERNEL, normalize_perf=False):
    # create a redis database
    database = redis.StrictRedis(host='localhost', port=6379)
    database.flushdb()
    
    # db_files = [f for f in glob(join(db_path, '*.db'))]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'merged' in f]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'merged' in f and 'updated3' in f]
    db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'v20' not in f and 'single-merged' in f and 'yunsheng' not in f]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'large-size' not in f and 'v20' in f]
    print(db_files)
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
        if keys[i][0:3] == 'lv1' or obj.ret_code.name == 'TIMEOUT' or obj.perf == 0:
            continue
        if keys[i][0:3] == 'lv1' and obj.perf == 0:
            count_lv1_invalid += 1
        if keys[i][0:3] == 'lv2' and obj.perf == 0:
            count_lv2_invalid += 1
        # if '__PARA__L0' in obj.point:
        #     print(obj.point)
        if keys[i][0:3] == 'lv2' and obj.point['__PIPE__L0'] == 'off' and obj.point['__PIPE__L1'] == 'off' and obj.point['__PIPE__L2'] == 'off' and obj.point['__PIPE__L3'] == 'off':
            count_filtered += 1
            print(keys[i], obj.perf)

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
                    # print(f'point exists with perf: {obj.perf}')
                    attrs = vars(obj)
                    # print(', '.join("%s: %s" % item for item in attrs.items()))
        # print(i)
        
        s = str(obj.point.values())
        all_points[(obj.perf, obj.ret_code.name + '-' + keys[i])] = (obj.point, obj.res_util, obj)
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
    print(count_filtered)
    # print(min(all_perfs))
    f_db.close()
    # print(len(set(points)), len(points))
    # print('zero_count', zero_count, 'nonzero_count', nonzero_count,
    #       zero_count + nonzero_count, 'count_lv1_invalid', count_lv1_invalid,
    #       'count_lv2_invalid', count_lv2_invalid, 'filtered:', count_filtered,
    #       'count_filtered_nonzero', count_filtered_nonzero)
    # print('perf_seen', perf_seen)



if __name__ == '__main__':
    for dataset in ['poly', 'machsuite']:
        if dataset == 'machsuite':
            db_path = f'{get_root_path()}/machsuite/databases/**/*'
            KERNELS = MACHSUITE_KERNEL
        elif dataset == 'poly':
            db_path = f'{get_root_path()}/poly/databases/**/*'
            KERNELS = poly_KERNEL
        for kernel in KERNELS:
            if '2mm' in kernel:
                print(f'now processing {kernel}')
                get_db_object(db_path, kernel)
            # break
    
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
