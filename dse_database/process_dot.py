from utils import save_pickle, load_pickle
import networkx as nx
from os.path import join, basename, splitext
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, Counter
import numpy as np
import logging


# dot_folder = '/Users/yba/Documents/UCLA/Research/software/dse_database/dotGenerator_all_kernels'
# cpp_folder = '/Users/yba/Documents/UCLA/Research/software/dse_database/ast_kernels'

# dot_folder = 'new_db/case2'
# cpp_folder = 'new_db/case2'

# dot_folder = '/home/yba/Documents/software/dse_database/machsuite/dot-files'
# cpp_folder = '/home/yba/Documents/software/dse_database/machsuite/sources'

# dot_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/new_db/case1'
# cpp_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/new_db/case1'

dot_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/machsuite/dot-files/'
cpp_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/machsuite/sources/'


all_cpp_files = sorted(glob(join(cpp_folder, '**', '*.cpp'), recursive=True))
print('Found', len(all_cpp_files), 'cpp/c files under', cpp_folder)

def preproc():
    for cpp_file in tqdm(all_cpp_files):
        # f = splitext(basename(cpp_file))[0]
        # f = 'svm'
        fn = basename(cpp_file)
        dot_file = join(dot_folder, f'{fn}.dot')
        g = nx.drawing.nx_pydot.read_dot(dot_file)
        save_pickle(g, f'{dot_file}.pickle', True)


def main():
    for cpp_file in tqdm(all_cpp_files):
        #preproc()
        _proc(cpp_file)

def _proc(cpp_file):
    f = basename(cpp_file).split('_')[0]
    fn = basename(cpp_file)
    dot_file = join(dot_folder, f'{fn}.dot')
    g = load_pickle(f'{dot_file}.pickle')
    g_new = nx.MultiDiGraph()
    lines = open(cpp_file).readlines()
    None_nodes = set()
    # print(g.number_of_nodes())
    for node, ndata in g.nodes(data=True):
        #print(ndata)
        g_new.add_node(node)
        assert 'label' in ndata, f'label not in ndata for {node}: {ndata}'
        nds = ndata['label'].split('\\n')
        attrs = {'shape': ndata.get('shape', '')}
        # try:
        attrs['type'] = nds[1] if len(nds) >= 2 else ''
        # except:
        #     print()
        content = _find_content(f, nds, lines)
        if content == 'None':
            None_nodes.add(node)
        attrs['content'] = content
        #print(content)
        nx.set_node_attributes(g_new, {node: attrs})
    for nid1, nid2, edata in g.edges(data=True):
        elabel = edata['label']
        g_new.add_edge(nid1, nid2, elabel=elabel)
    new_gexf_file = join(dot_folder, f'{fn}.gexf')

    # nx.write_gexf(g_new, join(dot_folder, f'{fn}_temp.gexf'))
    # exit()
    # print(g_new.number_of_nodes())
    while True:
        remove_nodes = set()
        for node in g_new.nodes():
            if len(list(g_new.neighbors(node))) == 0 and node in None_nodes:
                remove_nodes.add(node)
        for node in remove_nodes:
            g_new.remove_node(node)
        if not remove_nodes:
            break

    if g_new.number_of_nodes() == 0:
        print(cpp_file)

    nx.write_gexf(g_new, new_gexf_file)
    print(f'{new_gexf_file} generated with {g_new.number_of_nodes()} nodes')


def _find_content(f, nds, lines):
    f = strip_(f)
    for s_id, s in enumerate(nds):
        if 'compilerGenerated' in s or 'NULL_FILE' in s or 'SgSourceFile' in s:
            return 'None'
        if 'physical line=' in s:
            if '.cpp' in s or '.c' in s:
                line_num = int(s.split('physical line=')[1].split(')')[0])
                begin = int(s.split('raw line:col=')[1].split(')')[0].split(':')[1])
                # try:
                end = int(nds[s_id + 1].split('raw line:col=')[1].split(')')[0].split(':')[1])
                # except:
                #     print(nds)
                #     exit(-1)
                line = lines[line_num - 1]
                content = line[begin - 1:end]
                # print()
                return content
    return 'None'

def strip_(s):
    return s.replace('_', '')

if __name__ == '__main__':
    # preproc() # generate pickles from dot files
    main() # generate gexf files from pickle files
