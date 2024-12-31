from subprocess import Popen, PIPE, TimeoutExpired
from sortedcontainers import SortedDict
import os, time, pickle, dataclasses
import numpy as np

from result import Result
from utils import dirname, get_root_path
from config import FLAGS
from RL import dse_utils


@dataclasses.dataclass
class RemotePars:
    push_path = f'{get_root_path()}/hls_local' + '/{}'

remote_pars = RemotePars


class PrivateHQ:
    def __init__(self, B: int, dataset: str, kernel_name: str, uid: int, db_id: int = 0, redis_port = 0):
        self.B = B; self.dataset = dataset; self.kernel_name = kernel_name; 
        self.uid = uid; self.db_id = db_id
        self.buf: SortedDict = SortedDict()
        self.keys = set()
        self.cnt_query = 0
        self.redis_port = redis_port
        
    def append(self, res: Result):
        assert res.point is not None
        _key = dse_utils.point_to_str(res.point)
        if _key in self.keys:    # Do not append data repeatedly
            return
        # if not isinstance(res.perf, float):
        #     print(res.perf, type(res.perf))
        self.buf[(float(res.perf), _key)] = res
        self.keys.add(_key)
        assert len(self.buf) <= self.B
            
    def clear(self):
        self.buf: SortedDict = SortedDict()
        self.keys = set()
            
    def query_batch_remote(self, timeout: int):   # timeout (min)
        def _run(_cmd: str, verbose = False, error_retry = True):
            p = Popen(_cmd, stdout=PIPE, stderr=PIPE, shell=True, executable='/bin/bash')
            # p = Popen(_cmd, stdout=PIPE, stderr=PIPE)
            _success = False
            for i in range(20):
                try:
                    outs, errs = p.communicate(timeout=30)
                except TimeoutExpired:
                    print(_cmd, f'timeout, retrying {i}')
                else: 
                    if errs == b'' or not error_retry:
                        _success = True
                        break
            if verbose:
                print('==========================')
                print('CMD:', _cmd)
                print('outs:', outs.decode().strip('\n'))
                print('errs:', errs.decode().strip('\n'))
                print('==========================')
                print('', flush=True)
            assert _success # quit the job if failed
            return outs, errs
        
        self.cnt_query += 1
        _save_dict = dict()
        for i in range(len(self.buf)):
            _idx = -(i + 1)
            _item = self.buf.peekitem(_idx)   # _item[0] is the original key, and _item[1] is its value
            print(f'Design {i}: {_item[0]}')
            _save_dict[_item[0]] = _item[1]
        print('')
        push_path = remote_pars.push_path.format(f'{self.uid}/{self.cnt_query}/{self.kernel_name}_candi.pickle')
        _local_pull_path = dirname(push_path)
        os.makedirs(dirname(push_path), exist_ok=True)
        with open(push_path, 'wb') as handle:
            pickle.dump(_save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        
        if outs != b'' or errs != b'':
            assert 0, 'Error sending pickle to remote'
        
        # conda_path = f'{dirname(get_root_path())}/miniconda3'
        conda_path = f'{dirname(get_root_path())}/anaconda3'
        _remote_run_cmd = (
            f'nohup python parallel_run_tool_dse.py --kernel {self.kernel_name}'
            f'  --benchmark {self.dataset} --root-dir {get_root_path()} --pickle_path {push_path}'
            f'  --redis_port {self.redis_port}'
            f'  --db_id {self.db_id} --version {FLAGS.v_db} --timeout {timeout} </dev/null >nohup.out 2>&1 &\n'
            f'echo $! > {push_path}/pid.nohup\n'
            f'cat {push_path}/pid.nohup'
        )
        outs, errs = _run(_remote_run_cmd, verbose=True)
        if errs != b'':
            assert 0, 'Run remote process failed'
        _pid = outs.decode().strip('\n')
        _remote_poll_cmd = f'ps -efax -o pid | grep -w {_pid}'
        print('================= HLS ==================')
        print('Poll CMD:', _remote_poll_cmd)
        print('Running HLS', end='', flush=True)
        _tic = time.time()
        _nxt_checkpoint = 0

        while True:
            outs, errs = _run(_remote_poll_cmd, verbose=False)
            if outs != b'':    # It is still running
                assert str(_pid) in outs.decode().strip('\n')
            else:              # It has finished running
                print(outs.decode())
                print(errs.decode())
                break
            print('.', end='', flush=True)
            time.sleep(60.0)              # Query every minute
            if (time.time() - _tic) / 60 > _nxt_checkpoint:
                print('')
                print(f'{_nxt_checkpoint} minutes passed')
                _nxt_checkpoint += 5      # Print every 5 minutes
                print('Running HLS', end = '', flush=True)
        print('')
        print('HLS done!')
        print('========================================')
        print('')
        
        return f'{push_path}/{self.kernel_name}_result_updated-this-round-0.db'
