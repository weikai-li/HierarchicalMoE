from genericpath import exists
from tempfile import mkstemp
import shutil
from os.path import join, basename, dirname, isfile, exists, abspath
from os import remove
from utils import get_root_path, get_subdir, create_dir_if_not_exists
from glob import iglob
from pprint import pprint
from subprocess import Popen, PIPE
import json
from collections import OrderedDict
import time


kernel_root_path = join(dirname(dirname(get_root_path())), 'lorenzo-data/selected-kernels')
work_dir_base = '/expr/simple-example'
param_file = 'param.h'
makefile_path = join(get_root_path(), 'Makefile')

PRECISION = ['int', 'long', 'float', 'double']
SIZE = [64, 128, 256, 512, 1024, 2048]

def get_c_kernel(base_dir):
    c_kernel = [f for f in iglob(join(base_dir, '**'), recursive=True) if f.endswith('.c') and '_tb' not in f and 'common' not in f]
    assert len(c_kernel) == 1
    return c_kernel[0]

def get_kernel_files(kernel_root_path):
    kernel_info = OrderedDict()
    possible_kernels = get_subdir(kernel_root_path)
    for p_dir in possible_kernels:
        if isfile(join(p_dir, param_file)): ## if the param file exists
            c_kernel = get_c_kernel(p_dir)
            kernel_info[basename(p_dir)] = (c_kernel, join(p_dir, param_file)) ## kernel file, param file
            
    return kernel_info

def modify_kernel(src_dir, kernel_name):
    orig_file = join(src_dir, f'{kernel_name}.c')
    fnew, abs_path = mkstemp()
    found_kernel = False
    with open(fnew, 'w') as fpnew:
        with open(orig_file) as fp:
            for line in fp:
                if line.startswith('void'):
                    found_kernel = True
                    fpnew.write('#pragma ACCEL kernel\n')
                fpnew.write(line)
                
    if not found_kernel:
        print('No kernels found for', kernel_name, 'at', src_dir)
        raise RuntimeError()
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)
    
def modify_params(src_dir, precision, size):
    orig_file = join(src_dir, 'param.h')
    fnew, abs_path = mkstemp()
    new_lines = {'LEN_1D': f'#define LEN_1D {size}\n', 
                 'LEN_2D': f'#define LEN_2D {size}\n',
                 'real_t': f'#define real_t {precision}\n'}
    with open(fnew, 'w') as fpnew:
        for n, line in new_lines.items():
            fpnew.write(line)
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)
    
def modify_makefile(src_dir, kernel_name):
    orig_file = join(src_dir, 'Makefile')
    fnew, abs_path = mkstemp()
    with open(fnew, 'w') as fpnew:
        with open(orig_file) as fp:
            for line in fp:
                fpnew.write(line.replace('${KERNEL_NAME}', kernel_name))
    
    shutil.copymode(orig_file, abs_path)
    shutil.copy(abs_path, orig_file)
    
#define LEN_1D 320
#define LEN_2D 0
#define real_t char
        

def get_src_dir(base_path, kernel_name, precision, size):
    src_dir = join(base_path, kernel_name, f'{precision}-{size}', 'xilinx_dse')
    return src_dir

def create_work_dir(base_path, kernel_name, files_path, precision, size):
    src_dir = get_src_dir(base_path, kernel_name, precision, size)
    if exists(src_dir):
        shutil.rmtree(src_dir)
    create_dir_if_not_exists(src_dir)
    for f in files_path:
        if f.endswith('.c'):
            dest = join(src_dir, f'{kernel_name}.c')
        else:
            dest = src_dir
        shutil.copy(f, dest)
    shutil.copy(makefile_path, src_dir)
        
    modify_kernel(src_dir, kernel_name)
    modify_params(src_dir, precision, size)
    modify_makefile(src_dir, kernel_name)
    
    return src_dir

def find_and_add(prev_option, find_s, add_s):
    index = prev_option.find(find_s)
    return prev_option[:index+2] + add_s + prev_option[index+2:]

def add_parallel_option(ds_info):    
    for key, value in ds_info["design-space.definition"].items():
        if 'PARA' in key:
            prev_option = ds_info["design-space.definition"][key]["options"]
            if '128' in prev_option:
                prev_option = find_and_add(prev_option, '32', ',64')
            elif '64' not in prev_option:
                prev_option = find_and_add(prev_option, '32', ',64,128')
            ds_info["design-space.definition"][key]["options"] = prev_option
                
def autodse_dsgen(src_dir, src_file, ds_info):
    kernel_with_ds = [f for f in iglob(join(src_dir, '*'), recursive=True) if basename(f).startswith('rose_merlinkernel_')][0]
    shutil.copy(kernel_with_ds, src_file)
    ds_info["search.algorithm.name"] = "bottleneck"
    ds_info['timeout.exploration'] = 1200 
    ds_info['timeout.hls'] = 80 
    ds_info['timeout.transform'] = 20 
    
def replace_config(f_path, ds_info):
    remove(f_path)
    with open(f_path, 'w') as f_ds:
        json.dump(ds_info, f_ds, indent=4)
    
def run_autodse(project_dir, kernel_name, modify_config = True):
    src_dir = 'xilinx_dse'
    kernel_path = f'xilinx_dse/{kernel_name}.c'
    work_dir = 'work_dir'
    f_config = f'xilinx_dse/ds_info.json'
    if modify_config:
        p = Popen(f'cd {join(project_dir, src_dir)} \n source /share/atefehSZ/env.sh \n /share/atefehSZ/merlin_docker/docker-run-gnn.sh ds_generator {basename(kernel_path)}', shell = True, stdout=PIPE)
        while p.poll() is None:
            # ret = p.poll()
            time.sleep(0.5)
        if p.returncode != 0:
            print(f'ds-gen had non-zero output')
            return False
        with open(join(project_dir, f_config), 'r') as f_ds:
            ds_info = json.load(f_ds)
            add_parallel_option(ds_info)
            autodse_dsgen(join(project_dir, src_dir), join(project_dir, kernel_path), ds_info)
        replace_config(join(project_dir, f_config), ds_info)
        print(f'Added candidate pragmas. Now running the DSE...')
        p2 = Popen(f'cd {project_dir} \n source /share/atefehSZ/env.sh \n /share/atefehSZ/merlin_docker/docker-run-gnn.sh dse {src_dir} {work_dir} {f_config} fast', shell = True, stdout=PIPE)
        # ret = p2.poll()
        # while p2.poll() is None:
        #     # ret = p2.poll()
        #     time.sleep(0.5)
        text = (p2.communicate()[0]).decode('utf-8')
        if 'Finished' not in text:
            print(f'error in the current process. DSE did not finish')
        if p2.returncode != 0:
            print(f'process had non-zero output')
            
        return True
        # p.wait()
    else:
        p = Popen(f'cd {project_dir} \n source /share/atefehSZ/env.sh \n /share/atefehSZ/merlin_docker/docker-run-gnn.sh autodse {src_dir} {work_dir} {kernel_path} fast', shell = True, stdout=PIPE)
        p.wait()

    
def gather_db(base_path, kernel_name, copy_base_path):
    db_base_path = join(copy_base_path, 'databases/v18/one-db', kernel_name)
    create_dir_if_not_exists(db_base_path)
    src_base_path = join(copy_base_path, 'sources', kernel_name)
    create_dir_if_not_exists(src_base_path)
    config_base_path = join(copy_base_path, 'config', kernel_name)
    create_dir_if_not_exists(config_base_path)
    for precision in PRECISION:
        for size in SIZE:
            src_dir = get_src_dir(base_path, kernel_name, precision, size)
            if exists(src_dir):
                work_dir = join(dirname(src_dir), 'work_dir')
                ## copy source file
                shutil.copyfile(join(src_dir, f'{kernel_name}.c'), 
                                join(src_base_path, f'{kernel_name}-{precision}-{size}_kernel.c'))
                ## copy config file
                shutil.copyfile(join(src_dir, 'ds_info.json'), 
                                join(config_base_path, f'{kernel_name}-{precision}-{size}_ds_config.json'))
                ## copy db file
                shutil.copyfile(join(work_dir, 'result.db'), 
                                join(db_base_path, f'{kernel_name}-{precision}-{size}_result_updated-0.db'))
                print('done', src_dir)
            else:
                pass
                # print('No', src_dir, 'exists')


if __name__ == "__main__":
    kernel_info = get_kernel_files(kernel_root_path)
    # pprint(kernel_info)
    merlin_aux_makefile = '/expr/mcc_common'
    if not exists(merlin_aux_makefile):
        shutil.copytree('/share/atefehSZ/merlin/mcc_common', merlin_aux_makefile)
    
    target_kernel = list(kernel_info.keys())
    dse_run = False
    for kernel in target_kernel:
        print('#######################')
        print(f'started {kernel}')
        # if exists(join(work_dir_base, kernel)):
        #     shutil.rmtree(join(work_dir_base, kernel))
        if dse_run:
            for precision in PRECISION:
                if '1d-conv' in kernel and precision != 'long':
                    continue
                for size in SIZE:
                    if 'unrolled' in kernel and size <= 128: # and (size <= 512):
                        continue
                    if '1d-conv' in kernel and size <= 512:
                        continue
                    print(f'Now processing {kernel} with size {size} and precision {precision}')
                    src_dir = create_work_dir(work_dir_base, kernel, kernel_info[kernel], precision, size)
                    run_autodse(dirname(src_dir), kernel)
                # break
        else:
            copy_base_path = join(dirname(abspath(__file__)), 'simple-program')
            base_path = '/expr/simple-example'
            gather_db(base_path, kernel, copy_base_path)
