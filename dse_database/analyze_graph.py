import networkx as nx
from glob import glob
from os.path import basename, join
from utils import get_root_path
from scipy.stats import gmean


MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil_stencil2d',
                    'nw', 'md', 'stencil-3d']

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium']
ALL_KERNEL = {'machsuite': MACHSUITE_KERNEL, 'poly': poly_KERNEL}

sum_orig, sum_hier = 0, 0 
count = 0
orig_paths, hier_paths = [], []

for BENCHMARK in ['poly', 'machsuite']:
    processed_gexf_folder = join(get_root_path(), f'programl/{BENCHMARK}/processed')
    for name in ALL_KERNEL[BENCHMARK]:
        gexf_file_orig = glob(join(processed_gexf_folder, f'{name}_processed_result.gexf'))[0]
        gexf_file_hier = glob(join(processed_gexf_folder, 'extended-pseudo-block-connected-hierarchy', f'{name}_processed_result.gexf'))[0]
        g_orig = nx.readwrite.gexf.read_gexf(gexf_file_orig)
        g_hier = nx.readwrite.gexf.read_gexf(gexf_file_hier)
        cur_orig_avg_path = nx.average_shortest_path_length(g_orig)
        cur_hier_avg_path = nx.average_shortest_path_length(g_hier)
        print(basename(gexf_file_orig), cur_orig_avg_path, cur_hier_avg_path)
        sum_orig += cur_orig_avg_path
        sum_hier += cur_hier_avg_path
        orig_paths.append(cur_orig_avg_path)
        hier_paths.append(cur_hier_avg_path)
        count += 1
        
print(f'avg total -> orig: {sum_orig/count}, hier: {sum_hier/count}, count: {count}')
print(f'double check avg total -> orig: {sum(orig_paths)/len(orig_paths)}, hier: {sum(hier_paths)/len(hier_paths)}, count: {len(orig_paths)}')
print(f'geo mean total -> orig: {gmean(orig_paths)}, hier: {gmean(hier_paths)}, count: {len(hier_paths)}')
        
    