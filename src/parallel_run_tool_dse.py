# from localdse.explorer.single_query import run_query
import pickle
import redis
from os.path import join, dirname, basename, exists
import os
import pwd
import argparse
from glob import iglob
import json
from copy import deepcopy
import shutil
import numpy
import torch, io

from config import FLAGS
from utils import get_ts, create_dir_if_not_exists, get_src_path, get_host, get_root_path
import time
from subprocess import Popen, DEVNULL, PIPE
from result import Result
import io
import os
import socket
import psutil

model_tag = 'post-gnn-3lp-3lm' # 'DAC22-dse' # 
VER = 'v18_class' # 'v20_freeze1'
class MyTimer():
    def __init__(self) -> None:
        self.start = time.time()
    
    def elapsed_time(self):
        end = time.time()
        minutes, seconds = divmod(end - self.start, 60)
        
        return int(minutes)

class Saver():
    def __init__(self, kernel):
        self.logdir = join(
            get_src_path(),
            'logs',
            # 'yunsheng', 'dac-short', 'yizhou-spread-ds', f'MAML9-20d-5t-wo-zscore-run_tool-class-off_{kernel}_{get_ts()}')
            # 'yunsheng', 'dac-short', 'gnn-dse', f'spread-ds-run_tool-class-on_{kernel}_{get_ts()}')
            # 'yunsheng', 'm-post-dac', 'sepPT', f'kmeansNN-10d-run_tool-class-on_{kernel}_{get_ts()}')
            # 'post-dac', 'sa', f'pure-model-wo-zscore-run_tool-class-off_{kernel}_{get_ts()}')
            # 'dac', 'double-check', f'dac-model-8r-6c_{kernel}_{get_ts()}')
            # 'auto-encoder', 'all-data-sepPT/round1/task-transfer/gradually/freeze1', 'norm-perf-edge-attr-True-gae-on-T-off-P-class-off', f'6r_{kernel}_{get_ts()}') 
            f'auto-encoder/iccad/{model_tag}/{VER}/dse_results', f'6r_{kernel}_{get_ts()}')
            # 'auto-encoder', 'all-data-sepPT/round10-22kernel/task-transfer/freeze5', 'norm-perf-edge-attr-True-gae-on-T-off-P-class-off', f'6r_{kernel}_{get_ts()}') 
            # 'auto-encoder', 'extended-graph-db/round8', 'norm-perf-edge-attr-True-gae-on-T-off-P-class-off', f'6r_{kernel}_{get_ts()}') 
            # 'auto-encoder', 'extended-graph-db/round3/all-data-sepPTB/extended-connected', 'norm-perf-edge-attr-True-gae-on-T-off-P-class-off', f'6r_{kernel}_{get_ts()}') 
            # 'auto-encoder', 'hierarchy/all-data-sepPTB-extended-connected/round10-22kernel/task-transfer', 'correct-graph-type_6L-norm-perf-edge-attr-True-position-on-gae-on-T-off-P-class-off', f'6r_{kernel}_{get_ts()}') 
            # 'dac', 'round1/task-transfer', 'correct-graph-type_6L-norm-perf-edge-attr-True-position-on-gae-off-T-off-P-class-off', f'6r_{kernel}_{get_ts()}') 
        create_dir_if_not_exists(self.logdir)
        self.timer = MyTimer()
        print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')
    
    def info(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] INFO: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] INFO: {s}\n')
        self.log_f.flush()
        
    def error(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] ERROR: {s}')
        if not hasattr(self, 'log_e'):
            self.log_e = self._open('error.txt')
        self.log_e.write(f'[{elapsed}m] ERROR: {s}\n')
        self.log_e.flush()
        
    def warning(self, s, silent=False):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] WARNING: {s}')
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write(f'[{elapsed}m] WARNING: {s}\n')
        self.log_f.flush()
        
    def debug(self, s, silent=True):
        elapsed = self.timer.elapsed_time()
        if not silent:
            print(f'[{elapsed}m] DEBUG: {s}')
        if not hasattr(self, 'log_d'):
            self.log_d = self._open('debug.txt')
        self.log_d.write(f'[{elapsed}m] DEBUG: {s}\n')
        self.log_d.flush()

def gen_key_from_design_point(point) -> str:

    return '.'.join([
        '{0}-{1}'.format(pid,
                         str(point[pid]) if point[pid] else 'NA') for pid in sorted(point.keys())
    ])

def kernel_parser() -> argparse.Namespace:
    """Parse user arguments."""

    parser_run = argparse.ArgumentParser(description='Running Queries')
    parser_run.add_argument('--kernel',
                        required=True,
                        action='store',
                        help='Kernel Name')
    parser_run.add_argument('--benchmark',
                        required=True,
                        action='store',
                        help='Benchmark Name')
    parser_run.add_argument('--root-dir',
                        required=True,
                        action='store',
                        default='.',
                        help='GNN Root Directory')
    parser_run.add_argument('--pickle_path',
                        required=True,
                        action='store',
                        help = 'Path to pickle')
    parser_run.add_argument('--redis_port',
                        type=int,
                        required=True,
                        action='store',
                        help='The port number for redis database')
    parser_run.add_argument('--db_id',
                        required=True,
                        action='store',
                        default='0',
                        help='The database id for parallel run')
    parser_run.add_argument('--version',
                        required=True,
                        action='store',
                        default='v18',
                        help='The version of the Xilinx tool')
    parser_run.add_argument('--server',
                        required=False,
                        action='store',
                        default=None,
                        help='The container ID')
    parser_run.add_argument('--timeout',
                            required=True,
                            action='store',
                            default=150,
                            help='Timeout')

    return parser_run.parse_args()
    
def persist(database, db_file_path, id=0) -> bool:
    #pylint:disable=missing-docstring

    dump_db = {
        key: database.hget(id, key)
        for key in database.hgetall(id)
    }
    # print(dump_db)
    with open(db_file_path, 'wb') as filep:
        pickle.dump(dump_db, filep, pickle.HIGHEST_PROTOCOL)
        filep.flush()
    return True

def update_best(result_summary, key, result):
    thresh = 0.80000001
    if result.perf > 16.0 and result.perf < result_summary['min_perf']:
        is_min = True
    else:
        return
    max_utils = {'BRAM': thresh, 'DSP': thresh, 'LUT': thresh, 'FF': thresh}
    utils = {k[5:]: max(0.0, u) for k, u in result.res_util.items() if k.startswith('util-')}
    valid = all([(utils[res])< max_utils[res] for res in max_utils])
    if valid:
        result_summary['min_perf'] = result.perf
        result_summary['key_min_perf'] = key
        result_summary[key] = deepcopy(result)

def check_port(host, port):
    s = socket.socket()
    try:
        s.connect((host, port))
        return True
    except:
        return False
    finally:
        s.close()

def close_port(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            print(f"Closing port {port} by terminating PID {conn.pid}")
            process = psutil.Process(conn.pid)
            process.terminate()

def run_procs(saver, procs, database, kernel, f_db_new, result_summary, server=None):
    saver.info(f'Launching a batch with {len(procs)} jobs')
    try:
        while procs:
            prev_procs = list(procs)
            procs = []
            for p_list in prev_procs:
                text = 'None'
                idx, key, p = p_list
                ret = p.poll()
                if ret is not None and ret != 0:
                    text = (p.communicate()[0]).decode('utf-8')
                    saver.info(f'Job with batch id {idx} has non-zero exit code: {ret}')
                    saver.debug('############################')
                    saver.debug(f'Recieved output for {key}')
                    saver.debug(text)
                    saver.debug('############################')
                # Finished and successful
                elif ret is not None:
                    text = (p.communicate()[0]).decode('utf-8')
                    saver.debug('############################')
                    saver.debug(f'Recieved output for {key}')
                    saver.debug(text)
                    saver.debug('############################')

                    if server is not None and ('u22' not in server):
                        q_result = pickle.load(open(f'localdse/kernel_results/{kernel}_{idx}_{server}.pickle', 'rb'))
                    else:
                        q_result = pickle.load(open(f'localdse/kernel_results/{kernel}_{idx}.pickle', 'rb'))

                    for _key, result in q_result.items():
                        pickled_result = pickle.dumps(result)
                        if 'lv2' in key:
                            database.hset(0, _key, pickled_result)
                            database.hset(1, _key, pickled_result)
                        saver.info(f'Performance for {_key}: {result.perf} with return code: {result.ret_code} and resource utilization: {result.res_util}')
                        update_best(result_summary, key, result)
                    persist(database, f_db_new)
                # Still running
                else:
                    procs.append([idx, key, p])
                time.sleep(10)
    except:
        saver.error(f'Failed to finish the processes')
        raise RuntimeError()


args = kernel_parser()
saver = Saver(args.kernel)
CHECK_EARLY_REJECT = False

src_dir = join(args.root_dir, 'dse_database/save/merlin_prj', f'{args.kernel}', 'xilinx_dse')
username = pwd.getpwuid(os.getuid())[0]
work_dir = join(f'/scratch/{username}/workd', f'{args.kernel}', 'work_dir')
f_config = join(args.root_dir, 'dse_database', args.benchmark, 'config', f'{args.kernel}_ds_config.json')
f_pickle_path = join(args.pickle_path, '**')

print([f for f in iglob(f_pickle_path, recursive=True)])
f_pickle_list = [f for f in iglob(f_pickle_path, recursive=True) if f.endswith('.pickle') and f'{args.kernel}_' in f]
assert len(f_pickle_list) == 1
f_pickle = f_pickle_list[0]
db_dir = join(args.root_dir, 'dse_database', args.benchmark, 'databases', '**')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(module, name)
        parsed_module = module.split('.')
        print(parsed_module)
        if 'src' in parsed_module:
            module = parsed_module[-1]
        print(module, name)
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
        
        
result_dict = CPU_Unpickler(open(f_pickle, 'rb')).load()
create_dir_if_not_exists(dirname(work_dir))
create_dir_if_not_exists(work_dir)

max_db_id = 15
min_db_id = -1
found_db = False
for i in range(max_db_id, min_db_id, -1):
    if args.version == 'v18':
        f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_updated-{i}.db' in f and ('v18' in f and 'one-db-extended-round12' in f) and 'large-size' not in f]
    elif args.version == 'v20':
        f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_updated-{i}.db' in f and ('v20' in f and 'one-db' in f) and 'large-size' not in f and 'round3' in f]
    elif args.version == 'v21':
        f_db_list = [f for f in iglob(db_dir, recursive=True) if f'{args.kernel}_result_updated-{i}.db' in f and ('v21' in f and 'one-db' in f) and 'large-size' not in f and 'round3' in f]
    else:
        raise NotImplementedError()
    if len(f_db_list) == 1:
        f_db = f_db_list[0]
        print(f_db)
        f_db_new = join(args.pickle_path, basename(f_db).replace(f'_updated-{i}', f'_updated-gae-on-{i}'))
        f_db_new_this_round = join(args.pickle_path, basename(f_db).replace(f'_updated-{i}', f'_updated-this-round-{i}'))
        create_dir_if_not_exists(dirname(f_db_new))
        found_db = True
        break

database = redis.StrictRedis(host='localhost', port=args.redis_port, db=int(args.db_id))
database.flushdb()
try:
    file_db = open(f_db, 'rb')
    data = pickle.load(file_db)
    database.hmset(0, data)
except:   # This is our case
    f_db = None
    f_db_new = join(args.pickle_path, f'{args.kernel}_result_updated-gae-on-0.db')
    f_db_new_this_round = join(args.pickle_path, f'{args.kernel}_result_updated-this-round-0.db')
    saver.info('No prior databases')

batch_num = 100
batch_id = 0
procs = []
saver.info(f"""processing {f_pickle} 
    from db: {f_db} and 
    updating to {f_db_new}""")
saver.info(f"total of {len(result_dict.keys())} solution(s)")

if args.version == 'v18':
    database.hset(0, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
    database.hset(1, 'setup', pickle.dumps({'tool_version': 'SDx-18.3'}))
elif args.version == 'v20':
    database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
    database.hset(1, 'setup', pickle.dumps({'tool_version': 'Vitis-20.2'}))
elif args.version == 'v21':
    database.hset(0, 'setup', pickle.dumps({'tool_version': 'Vitis-21.1'}))
    database.hset(1, 'setup', pickle.dumps({'tool_version': 'Vitis-21.1'}))
else:
    raise NotImplementedError()
    
min_perf = float("inf")
result_summary = {'key_min_perf': None, 'min_perf': float("inf")}
hls_result_dict = {}
for result_key, result in sorted(result_dict.items()):
    if hasattr(result, 'point'):
        point_ = result.point
    else:
        point_ = result
    for key_, value in point_.items():
        if type(value) is str or type(value) is int:
            pass
        else:
            point_[key_] = value.item()
    key = f'lv2:{gen_key_from_design_point(point_)}'
    lv1_key = key.replace('lv2', 'lv1')
    isEarlyRejected = False
    rerun = False
    if CHECK_EARLY_REJECT and database.hexists(0, lv1_key):   # CHECK_EARLY_REJECT = False
        pickle_obj = database.hget(0, lv1_key)
        obj = pickle.loads(pickle_obj)
        if obj.ret_code.name == 'EARLY_REJECT':
            isEarlyRejected = True
    
    if database.hexists(0, key):   # In our case, this is False
        # print(f'key exists {key}')
        pickled_obj = database.hget(0, key)
        obj = pickle.loads(pickled_obj)
        if obj.perf == 0.0:
            # print(f'should rerun for {key}')
            rerun = True

    if rerun or (not isEarlyRejected and not database.hexists(0, key)):  # This is our case
        hls_result_dict[result_key] = result
        pass
    elif isEarlyRejected:
        pickled_obj = database.hget(0, lv1_key)
        obj = pickle.loads(pickled_obj)
        result.actual_perf = 0
        result.ret_code = Result.RetCode.EARLY_REJECT
        result.valid = False
        saver.info(f'LV1 Key exists for {key}, EARLY_REJECT')
    else:
        pickled_obj = database.hget(0, key)
        obj = pickle.loads(pickled_obj)
        result.actual_perf = obj.perf
        saver.info(f'Key exists. Performance for {key}: {result.actual_perf} with return code: {result.ret_code} and resource utilization: {obj.res_util}')
        update_best(result_summary, key, obj)
        database.hset(1, key, pickled_obj)

    
if len(hls_result_dict) == 0:
    persist(database, f_db_new_this_round, id=1)

port = args.redis_port + 20
while check_port('localhost', port) == True:
    port += 1
print(f'using port {port} to run HLS')

for _, result in sorted(hls_result_dict.items()):
    if hasattr(result, 'point'):
        point_ = result.point
    else:
        point_ = result
    if 'ellpack' in args.kernel:
        break
    if len(procs) == batch_num:   # procs = []
        run_procs(saver, procs, database, args.kernel, f_db_new, result_summary, args.server)
        batch_id == 0
        procs = []
    for key_, value in point_.items():
        if type(value) is str or type(value) is int:
            pass
        else:
            point_[key_] = value.item()
    key = f'lv2:{gen_key_from_design_point(point_)}'
    lv1_key = key.replace('lv2', 'lv1')
    isEarlyRejected = False
    rerun = False
    if CHECK_EARLY_REJECT and database.hexists(0, lv1_key):
        pickle_obj = database.hget(0, lv1_key)
        obj = pickle.loads(pickle_obj)
        if obj.ret_code.name == 'EARLY_REJECT':
            isEarlyRejected = True
    
    if database.hexists(0, key):
        # print(f'key exists {key}')
        pickled_obj = database.hget(0, key)
        obj = pickle.loads(pickled_obj)
        if obj.perf == 0.0:
            # print(f'should rerun for {key}')
            rerun = True
    
    assert rerun or (not isEarlyRejected and not database.hexists(0, key))

    kernel = args.kernel
    if args.server is not None and ('u22' not in args.server):
        result_file = f'./localdse/kernel_results/{args.kernel}_point_{batch_id}_{args.server}.pickle'
    else:   # This is our case
        result_file = f'./localdse/kernel_results/{args.kernel}_point_{batch_id}.pickle'
    with open(result_file, 'wb') as handle:
        pickle.dump(point_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    new_work_dir = join(work_dir, f'batch_id_{batch_id}')
    if args.version == 'v18':
        env, docker = 'env', 'docker'
    elif args.version == 'v20':
        env, docker = 'vitis_env', 'docker'
    elif args.version == 'v21':
        pass
    else:
        raise NotImplementedError()
    if args.version == 'v21':
        # Y: even though the point is early rejected, run HLS
        # N: if early rejected, do not run HLS !!!warning!!!!
        run_er = 'Y'
        p = Popen(f'module load apptainer\n source {get_root_path()}/hls_sh/vitis_env.sh\n apptainer run --bind /opt/xilinx,/scratch/{username},/mnt,/tmp/home_{username}:/tmp/home {get_root_path()}/src/{FLAGS.merlin_path} bash {get_root_path}/hls_sh/run.sh -p {port} -c \"python3 -m autodse.explorer.single_query --src-dir {src_dir} --work-dir {new_work_dir} --kernel {kernel} --config {f_config} --id {batch_id} --timeout {args.timeout} --port {port} --run_er {run_er} \"', shell = True, stdout=PIPE, executable='/bin/bash')
    elif args.server is not None and 'u22' in args.server:
        p = Popen(f"cd {get_src_path()} \n source {share}/{env}.sh \n {share}/merlin_docker/{docker}-run-gnn-new.sh -s python3 -m autodse.explorer.single_query --src-dir {src_dir} --work-dir {new_work_dir} --kernel {kernel} --config {f_config} --id {batch_id} --timeout {args.timeout}", shell = True, stdout=PIPE)
    else:
        p = Popen(f"cd {get_src_path()} \n source {share}/{env}.sh \n {share}/merlin_docker/{docker}-run-gnn-new.sh -s python3 -m autodse.explorer.single_query --src-dir {src_dir} --work-dir {new_work_dir} --kernel {kernel} --config {f_config} --id {batch_id} --timeout {args.timeout}", shell = True, stdout=PIPE)
    
    procs.append([batch_id, key, p])
    saver.info(f'Added {point_} with batch id {batch_id}')
    batch_id += 1

if len(procs) > 0:
    run_procs(saver, procs, database, args.kernel, f_db_new, result_summary, args.server)
persist(database, f_db_new_this_round, id=1)

if result_summary['key_min_perf'] != None:
    saver.info('#####################################')
    key = result_summary['key_min_perf']
    result = result_summary[key]
    saver.info(f'Min perf is {result.perf} for {key} with return code: {result.ret_code} and resource utilization: {result.res_util}')
else:
    saver.info('#####################################')
    saver.info('No valid point generated.')
    
try:
    file_db.close()
except:
    print('file_db is not defined')

close_port(port)
