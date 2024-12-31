from utils import save_pickle, load_pickle
import networkx as nx
from os.path import join, basename, splitext
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, Counter
import numpy as np
import logging
import os
import pickle
import json
import ast
from copy import deepcopy


# dot_folder = '/Users/yba/Documents/UCLA/Research/software/dse_database/dotGenerator_all_kernels'
# cpp_folder = '/Users/yba/Documents/UCLA/Research/software/dse_database/ast_kernels'

# dot_folder = 'new_db/case2'
# cpp_folder = 'new_db/case2'
# dot_folder = 'programl_graphs'
# cpp_folder = 'ast_kernels'

# raw_gexf_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/programl/machsuite/ncubed/'
cpp_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/programl/poly/'
raw_gexf_folder = cpp_folder
processed_gexf_folder = '/home/atefehSZ/RL/original-software-gnn/software-gnn/dse_database/programl/poly/processed/'
KERNEL= 'bicg'



all_cpp_files = sorted(glob(join(cpp_folder, '**', '*.cpp'), recursive=True))
all_cpp_files += sorted(glob(join(cpp_folder, '**', '*.c'), recursive=True))
print('Found', len(all_cpp_files), 'cpp/c files under', cpp_folder)

def preproc():
    for cpp_file in tqdm(all_cpp_files):
        # f = splitext(basename(cpp_file))[0]
        # f = 'svm'
        f = basename(cpp_file).split('.')[0]
        fn = basename(cpp_file)
        #dot_file = join(dot_folder, f'{f}_graph.dot')
        gexf_file = join(raw_gexf_folder, f'{f}.gexf')
        if os.path.exists(gexf_file):
            # g = nx.drawing.nx_pydot.read_dot(dot_file)
            g = nx.readwrite.gexf.read_gexf(gexf_file)
            with open(f'{gexf_file}.pickle', 'wb') as handle:
                pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('wrote pickle')


def main():
    for cpp_file in tqdm(all_cpp_files):
        if KERNEL in cpp_file:
            _proc(cpp_file)

def _proc(cpp_file):
    f = basename(cpp_file).split('.')[0]
    fn = basename(cpp_file)
    # dot_file = join(dot_folder, f'{f}_graph.dot')
    gexf_file = join(raw_gexf_folder, f, f'{f}.gexf')
    print(gexf_file)
    if os.path.exists(gexf_file):
        print(gexf_file)
        # g = load_pickle(f'{gexf_file}.pickle')
        g = nx.readwrite.gexf.read_gexf(gexf_file)
        g_new = nx.MultiDiGraph()
        for node, ndata in g.nodes(data=True):
            attrs = deepcopy(ndata)
            if 'features' in ndata:
                feat = ast.literal_eval(ndata['features'])
                attrs['full_text'] = feat['full_text'][0]
                del attrs['features']
                
            g_new.add_node(node)
            #print(node, attrs, ndata)
            nx.set_node_attributes(g_new, {node: attrs})
            
             
        #     assert 'label' in ndata, f'label not in ndata for {node}: {ndata}'
        #     nds = ndata['label'].split('\\n')
        #     attrs = {'shape': ndata.get('shape', '')}
        #     # try:
        #     attrs['type'] = nds[1] if len(nds) >= 2 else ''
        #     # except:
        #     #     print()
        #     content = _find_content(f, nds, lines)
        #     if content == 'None':
        #         None_nodes.add(node)
        #     attrs['content'] = content
        #     # print(content)
        edge_list = []  
        for nid1, nid2, edata in g.edges(data=True):
            edge_list.append((nid1, nid2, edata))
            if edata['flow'] == 3:
                edge_list.append((nid2, nid1, edata))
                #print(nid1, nid2, edata)
        g_new.add_edges_from(edge_list)
        # for nid1, nid2, edata in g_new.edges(data=True):
        #     print(nid1, nid2, edata)
            #i += 1
        
            #elabel = edata['label']
            #g_new.add_edge(nid1, nid2, edata)
        new_gexf_file = join(processed_gexf_folder, f'{f}_processed_result.gexf')

    # nx.write_gexf(g_new, join(dot_folder, f'{fn}_temp.gexf'))
    # exit()
        # print([n for n in g_new.neighbors('23')])
        # print([n for n in g_new.neighbors('165')])
        while True:
            remove_nodes = set()
            for node in g_new.nodes():
                if len(list(g_new.neighbors(node))) == 0 or node is None:
                    print(node)
                    remove_nodes.add(node)
            for node in remove_nodes:
                g_new.remove_node(node)
            if not remove_nodes:
                break

        # if g_new.number_of_nodes() == 0:
        #     print(cpp_file)

        print(len(g_new.nodes), len(g.nodes))
        print(len(g_new.edges), len(g.edges))
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
    # preproc() # generate pickles
    main()
