import networkx as nx
from os.path import join
from glob import glob
from tqdm import tqdm
from collections import OrderedDict, Counter
import numpy as np
import logging

logging.disable(logging.CRITICAL)
# logging.disable(logging.NOTSET)


PATH = '/home/yba/Documents/software/dse_database/dotGenerator_all_kernels/'
FILES = glob(join(PATH, '**', '*.dot'), recursive=True)
print('Found', len(FILES), 'dot under', PATH)


# FILES = FILES[0:1]

def _print(nns, nes, etypes):
    _print_stats(nns, '#nodes')
    _print_stats(nes, '#edges')
    print('etypes', len(etypes), etypes.most_common(10))


def _print_stats(li, name):
    stats = OrderedDict()
    stats['#'] = len(li)
    stats['Avg'] = np.mean(li)
    stats['Std'] = np.std(li)
    stats['Min'] = np.min(li)
    stats['Max'] = np.max(li)
    print(name)
    for k, v in stats.items():
        print(f'\t{k}:\t{v}')


nns = []
nes = []
etypes = Counter()
for file in tqdm(FILES):
    # print()

    g = nx.drawing.nx_pydot.read_dot(file)

    # print(type(x))

    edges = g.edges(data='Label')
    ne = len(g.edges())
    etypes.update([e[1] for e in edges])
    nn = g.number_of_nodes()
    nns.append(nn)
    nes.append(ne)
    print(f'nn={nn} ne={ne}')
    _print(nns, nes, etypes)

_print(nns, nes, etypes)
