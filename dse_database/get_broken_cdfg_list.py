from utils_old import save_pickle, load_pickle
import networkx as nx
from os.path import join, basename, splitext, dirname
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, Counter
import numpy as np
import logging
# dot_folder = '/Users/yba/Documents/UCLA/Research/software/dse_database/dotGenerator_all_kernels'
# cpp_folder = '/Users/yba/Documents/UCLA/Research/software/dse_database/ast_kernels'
cdfg_folder = 'cdfg'
all_cdfg_files = sorted(glob(join(cdfg_folder, '**', '*.dot'), recursive=True))
ouput_list_save_path = '/home/vagrant/list_broken_cdfg.txt'
print('Found', len(all_cdfg_files), 'cdfg files under', cdfg_folder)
def preproc():
    log = open(ouput_list_save_path, 'w')
    for dot_file in tqdm(all_cdfg_files):
        print(dot_file)
        try:
            g = nx.drawing.nx_pydot.read_dot(dot_file)
        except:
            print('error parsing {}'.format(dot_file))
            log.write(dot_file)
            log.write('\n')
    log.close()
def main():
    for cpp_file in tqdm(all_cpp_files):
        _proc(cpp_file)
def _proc(cpp_file):
    f = basename(cpp_file).split('_')[0]
    fn = basename(cpp_file)
    dot_file = join(dot_folder, f'{fn}.dot')
    g = load_pickle(f'{dot_file}.pickle')
    g_new = nx.MultiDiGraph()
    lines = open(cpp_file).readlines()
    None_nodes = set()
    for node, ndata in g.nodes(data=True):
        # print(ndata)
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
        # print(content)
        nx.set_node_attributes(g_new, {node: attrs})
    for nid1, nid2, edata in g.edges(data=True):
        elabel = edata['label']
        g_new.add_edge(nid1, nid2, elabel=elabel)
    new_gexf_file = join(dot_folder, f'{fn}.gexf')
    # nx.write_gexf(g_new, join(dot_folder, f'{fn}_temp.gexf'))
    # exit()
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
def _find_content(f, nds, lines):
    f = strip_(f)
    for s_id, s in enumerate(nds):
        if 'compilerGenerated' in s or 'NULL_FILE' in s or 'SgSourceFile' in s:
            return 'None'
        if 'physical line=' in s:
            if '.cpp' in s:
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
    preproc() # generate pickles
    # main()