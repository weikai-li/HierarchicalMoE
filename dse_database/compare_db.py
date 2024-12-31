import redis
import pickle
from collections import Counter
from os.path import join
from glob import glob, iglob


# DATASET = 'cnn1'
# DATASET = 'machsuite'
DATASET = 'poly'
if DATASET == 'cnn1':
    DIR = './save/cnn1'
    KERNEL = 'cnn'
    gexf_file = './new_db/case1/CnnKernel1.cpp.gexf'
    db_path = './databases/cnn_case1/'
elif DATASET == 'machsuite':
    DIR = './save/machsuite/gemm-ncubed'
    KERNEL = 'gemm-ncubed'
    gexf_file = './machsuite/dot-files/gemm-ncubed_kernel.cpp.gexf'
    db_path = './machsuite/databases/**/*'
    # db_path = './machsuite/databases/original-size/bottleneck/**/*'
elif DATASET == 'poly':
    db_path = './poly/databases/**/*'



def init_db():
    database = redis.StrictRedis(host='localhost', port=6379)
    database.flushdb()
    return database
    
def get_db_files(db_path, key_word1 = None, key_word2 = None):
    db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if (key_word1 is None or key_word1 in f)
                and (key_word2 is None or key_word2 in f) and ('archive' not in f)]
    # db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True) if 'archive' in f]
    return db_files

def read_db(database, db_files, combine = True, idx_db = 0):
    '''
    combine: merge all the databases as one or read them separately
    '''
    # idx_db = 0
    for idx, file in enumerate(db_files):
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        database.hmset(idx_db, data)
        if not combine:
            idx_db += 1
        f_db.close()
    if combine: idx_db = 1
    return idx_db
            
def get_keys(database, num_db):
    all_keys = {}
    for n in range(num_db):
        keys = [k.decode('utf-8') for k in database.hkeys(n)]
        all_keys[n] = sorted(keys)
    return all_keys

def compare_points_and_keys(all_points, all_keys, first, second): 
    res = (all_points[first]).items() <= (all_points[second]).items()
    res_key = set(all_keys[first]).issubset(set(all_keys[second]))
    
    return res, res_key 
        

# run "redis-server" on command line first!
def compare_db_object(db_path, key_word1 = None, key_word2 = None, normalize_perf=False):
    # create a redis database
    database = init_db()
    
    db_files = get_db_files(db_path, key_word1, key_word2)
    print(db_files)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    num_db = read_db(database, db_files, combine = False)
    max_idx = len(db_files)
    all_keys = get_keys(database, num_db)
    print(num_db)

    
    all_points = {}
    all_perf_seen = {}
    all_perfs = {}
    iii = 0
    for id, keys in all_keys.items():
        print(f'number of keys for database {id}: {len(keys)}')
        points = {}
        perf_seen = Counter()
        perfs = []
        diff_label = 0
        for i in range(len(keys)):
            pickle_obj = database.hget(id, keys[i])
            if pickle_obj is None:
                print(f'key {i}:{keys[i]} yeilds none object')
                continue
            obj = pickle.loads(pickle_obj)
            if type(obj) is int or type(obj) is dict:
                continue
            # if 'lv1' in keys[i]: #and obj.ret_code.name == 'PASS':
            #     continue
            iii += 1    
            # print(i)
            s = str(obj.point.values())
            if s not in points:
                points[s] = (keys[i], obj.perf, obj.ret_code.name)
            else:
                diff_label_bool = obj.perf != points[s][2]
                if diff_label_bool:
                    diff_label += 1
            perf_seen[obj.perf] += 1
            perfs.append(obj.perf)
        all_points[id] = points
        all_perf_seen[id] = perf_seen
        all_perfs[id] = perfs
        print(perf_seen)
        print(iii)
        print(f'max cycle count for {id} is {max(perf_seen.keys())} and min cycle count is {min(perf_seen.keys())}')
    
    first = 0; second = 1
    if (len(all_keys.keys()) > 1):
        if len(all_keys[0]) > len(all_keys[1]):
            first = 1; second = 0
        res, res_key = compare_points_and_keys(all_points, all_keys, first, second)    
        print(f'all points of db {first} exits in db {second}: {res}')
        print(f'all keys of db {first} exits in db {second}: {res_key}')
        
        for idx, p in all_points[first].items():
            if idx not in all_points[second]:
                print(f'found a point that does not exist in {second}')
                break
            else:
                if p != all_points[second][idx]:
                    print(f'not the same {p} vs {all_points[second][idx]}')
                    #break



if __name__ == '__main__':
    compare_db_object(db_path, 'mvt', 'bottleneck')
 

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
