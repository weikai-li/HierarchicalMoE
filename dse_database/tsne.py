from utils_old import get_root_path, MLP, print_stats
import os.path as osp
from os.path import join, basename
from glob import glob, iglob
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import networkx as nx
import redis, pickle, random
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack

from tqdm import tqdm

from shapely.geometry import Point, LineString

from utils import save_fig, create_dir_if_not_exists
import pickle

class LineInterp(object):
    def __init__(self, x1, y1, x2, y2, num_points):
        assert (num_points >= 2)
        self.x1, self.y1, self.x2, self.y2, self.num_points = \
            x1, y1, x2, y2, num_points
    def name(self):
        return 'line_interp_x1={}_y1={}_x2={}_y2={}_{}ps'.format(
            self.x1, self.y1, self.x2, self.y2, self.num_points)

confs = [LineInterp(x1=7.5, y1=-6, x2=9, y2=-14, num_points=6)]
COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
PERF = 'LUT'
PERF_all = ['perf', 'LUT', 'DSP', 'FF', 'BRAM']
#PERF_all = ['perf']
perplexity = 25
n_iter = 5000
VER = 'v18'
if VER == 'v18': nonver = 'v20'
elif VER == 'v20': nonver = 'v18'
COMMON = False

GEXF_FOLDER = join(get_root_path(), 'programl', '**', 'processed', '**')
print(GEXF_FOLDER)
# DATASET = 'cnn1'
DATASET = 'machsuite'
if DATASET == 'cnn1':
    DIR = './save/cnn1'
    KERNEL = 'cnn'
    gexf_file = './new_db/case1/CnnKernel1.cpp.gexf'
    db_path = './databases/cnn_case1/'
elif DATASET == 'machsuite':
    DIR = './save/machsuite/gemm-ncubed'
    KERNEL = 'bicg'
    # KERNEL = 'stencil'
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and KERNEL in f])
    print(GEXF_FILES)
    gexf_file = GEXF_FILES[0]
    db_path = './machsuite/databases/**/*'
    db_path = './poly/databases/**/*'

def vis_data():
    g = nx.read_gexf(gexf_file)

    database = redis.StrictRedis(host='localhost', port=6379)

    enc_ptype = OrdinalEncoder()

    X, y = [], []

    database.flushdb()
    # open a sample database
    # f_db = open(db_path, "rb")

    # db_files = [f for f in iglob(join(db_path, '*bottleneck*.db'), recursive=True)]
    # db_files = [f for f in iglob(join(db_path, '*.db'), recursive=True)]
    # db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and 'original-size' in f and 'v18' in f and 'one-db' in f]
    db_files = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and VER in f and 'one-db' in f]
    print(db_files)
    if COMMON:
        db_files_other = [f for f in iglob(db_path, recursive=True) if f.endswith('.db') and KERNEL in f and nonver in f and 'one-db' in f]
        print(db_files_other)
    # load the database and get the keys
    # the key for each entry shows the value of each of the pragmas in the source file
    for idx, file in enumerate(db_files):
        f_db = open(file, 'rb')
        data = pickle.load(f_db)
        database.hmset(0, data)
        max_idx = idx + 1

    if COMMON: 
        for idx, file in enumerate(db_files_other):
            f_db = open(file, 'rb')
            data = pickle.load(f_db)
            database.hmset(1, data)
            max_idx = idx + 1
    # data = pickle.load(f_db)
    # database.hmset(0, mapping=data)
    keys = [k.decode('utf-8') for k in database.hkeys(0)]

    keys = sorted(keys)
    print(g.number_of_nodes())
    if COMMON:
        y_dict = {}
        for objective in PERF_all:
            y_dict[f'{objective}-{VER}'] = []
            y_dict[f'{objective}-{nonver}'] = []
    else:
        y_dict = {objective: [] for objective in PERF_all}
    for i in range(len(keys)):
        if keys[i][0:3] == 'lv1':
            continue
        pickle_obj = database.hget(0, keys[i])
        obj = pickle.loads(pickle_obj)
        if type(obj) is int or type(obj) is dict:# or obj.perf == 0:
            continue
        # if keys[i][0:3] == 'lv1' and obj.ret_code.name == 'PASS':
        if obj.perf == 0.0:
            continue
        if COMMON:
            if not database.hexists(1, keys[i]):
                continue
            else:
                obj2 = pickle.loads(database.hget(1, keys[i]))
        # if obj.perf == 0:
        #     y.append(0)
        # else:
        #     y.append(1)
        for objective in PERF_all:
            if objective == 'perf':
                if COMMON:
                    y_dict[f'{objective}-{VER}'].append(obj.perf)
                    y_dict[f'{objective}-{nonver}'].append(obj2.perf)
                else:
                    y_dict[objective].append(obj.perf)
            else:
                if COMMON:
                    y_dict[f'{objective}-{VER}'].append(obj.res_util[f'util-{objective}'])
                    y_dict[f'{objective}-{nonver}'].append(obj2.res_util[f'util-{objective}'])
                else:
                    y_dict[objective].append(obj.res_util[f'util-{objective}'])
        X_ptypes = []
        for node, ndata in g.nodes(data=True):
            # print(node['type'], type(node['type']))
            if 'full_text' in ndata and 'pragma' in ndata['full_text']:
                p_text = ndata['full_text'].rstrip()
                assert p_text[0:8] == '#pragma '
                t = p_text[8:].upper()
                # print(t)
                for k in obj.point:
                    kl = 'AUTO{' + k + '}'
                    if kl in t:
                        t = t.replace(kl, str(obj.point[k]))
                        X_ptypes.append(t)
                        # print(t)

                    # t = ' '.join(t.split(' ')[0:2])

            # else:
            # ptype = 'None'
        X.append(X_ptypes)
        #print(X_ptypes)

    f_db.close()
    if not COMMON:
        for objective in PERF_all:
            print(f'number of classes for {objective} {len(set(y_dict[objective]))}')
            print(f'{objective}: nonzero={np.count_nonzero(y_dict[objective])} ({np.count_nonzero(y_dict[objective]) / len(y_dict[objective])}) '
            f'zero={len(y_dict[objective]) - np.count_nonzero(y_dict[objective])} tot={len(y_dict[objective])}')
    X_old = X
    # print(X)
    print(len(X), len(X[0]))
    X = enc_ptype.fit_transform(X_old)
    # print(X)
    print(X.shape[0], X.shape[1])

    feat_cols = ['feat' + str(i) for i in range(X.shape[1])]
    print(feat_cols)
    for objective in PERF_all:
        if COMMON:
            for v in [VER, nonver]:
                df = pd.DataFrame(X, columns=feat_cols)
                df['y'] = y_dict[f'{objective}-{v}']
                print('Size of the dataframe: {}'.format(df.shape))
                embs = tsne(df, feat_cols, len(set(y_dict[f'{objective}-{v}'])), f'{objective}-{v}')
            
        else:
            df = pd.DataFrame(X, columns=feat_cols)
            df['y'] = y_dict[objective]
            
            # df['label'] = df['y'].apply(lambda i: str(i))
            print('Size of the dataframe: {}'.format(df.shape))

            embs = tsne(df, feat_cols, len(set(y_dict[objective])), objective)
    
    return embs, X_old

    # from sklearn.svm import SVC
    # from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    # from sklearn.neural_network import MLPClassifier
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=.4, random_state=42)

    # for clf, name in [(AdaBoostClassifier(), 'AdaBoostClassifier'),
    #                   (SVC(kernel="linear", C=0.025), 'Linear SVM'),
    #                   (SVC(gamma=2, C=1), 'RBF SVM'),
    #                   (RandomForestClassifier(), 'RandomForestClassifier'),
    #                   (MLPClassifier(), 'MLPClassifier')]:
    #     print(name)
    #     clf.fit(X_train, y_train)
    #     for X, y, label in [(X_train, y_train, 'train'),
    #                         (X_test, y_test, 'test')]:
    #         y_pred = clf.predict(X)
    #         acc = accuracy_score(y_true=y, y_pred=y_pred)
    #         print(label, 'X.shape', X.shape, 'len(y)', len(y), 'acc=', acc)


def tsne(df, feat_cols, n_class, obj = None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    X_to_fit = df[feat_cols].values
    df.to_pickle('test_pickle.pkl')
    print(f'Size of X_to_fit: {X_to_fit.shape}')
    tsne_results = tsne.fit_transform(X_to_fit)
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    cmap = sns.color_palette("icefire", as_cmap=True)
    sns_plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=cmap,
        data=df,
        legend="full",
        # alpha=0.3
    )
    sns_plot.set(title=f'{obj}')
    if obj == 'perf':
        # sns_plot.legend(loc='center left',  ncol=4)
        # sns_plot.legend(loc='upper right', bbox_to_anchor=(0.0, 0.5), ncol=2)
        sns_plot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=5)
        # sns_plot.subplots_adjust(right=0.7)
    if COMMON:
        subdir = 'common'
    else:
        subdir = VER
    create_dir_if_not_exists(join(get_root_path(), DIR, KERNEL, subdir))
    save_path = join(get_root_path(), DIR, KERNEL, subdir, f'{obj}-tsne-p{perplexity}-n{n_iter}.png')
    sns_plot.get_figure().savefig(save_path, bbox_inches='tight')
    save_path_npy = join(get_root_path(), DIR, KERNEL, subdir, f'{obj}-tsne-p{perplexity}-n{n_iter}.npy')
    np.save(save_path_npy, tsne_results)
    
    return tsne_results


def _to_points(embs):
    assert (embs.shape[1] == 2)
    rtn = []
    for emb in embs:
        rtn.append(Point(emb))
    print('Created {} Point objects'.format(len(rtn)))
    return rtn


def _plot_conf(points, conf, X_pragmas, y):
    line = LineString([(conf.x1, conf.y1), (conf.x2, conf.y2)])
    print(line)
    selected_ps = []
    selected_ids = []
    for perc in _get_percs(conf):
        lp = line.interpolate(perc, normalized=True)
        id, emb_p = _closest_point(lp, points)
        selected_ps.append(emb_p)
        selected_ids.append(id)
    print('plotting...')
    for i in selected_ids:
        print(points[i], y[i], X_pragmas[i])
    _plot_points(points, [], conf)
    _plot_points(points, selected_ps, conf)
    #_plot_gs(selected_gs, conf)
    
    
def _plot_gs(selected_gs, conf):
    assert (len(selected_gs) >= 2)
    info_dict = {
        # draw node config
        'draw_node_size': 20,
        'draw_node_label_enable': True,
        'show_labels': False,
        'node_label_type': 'type',
        'node_label_name': 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': get_color_map(selected_gs),
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 10,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    plt_cnt = 0
    info_dict, plt_cnt = set_save_paths_for_vis(
        info_dict, DIR, None, 'binary-{}_{}_gs'.format(DATASET, conf.name()), plt_cnt)
    info_dict['each_graph_text_list'] = [i + 1 for i in range(len(selected_gs))]
    vis_small(selected_gs[0], selected_gs[1:], info_dict)
    print('Saved {} query demo plots'.format(plt_cnt))
    
    
def _plot_points(points, selected_ps, conf):
    plt.figure()
    # for p in points:
    #     x = _get_coord(p)[0]
    #     y = _get_coord(p)[1]
    #     plt.scatter(x, y, c=COLOR_P, marker='*')
    for i, p in enumerate(selected_ps):
        x = _get_coord(p)[0]
        y = _get_coord(p)[1]
        plt.scatter(x, y, facecolors='none', edgecolors=COLOR_M, s=200)
        # if i == 0 or i == len(selected_ps) - 1:
        plt.annotate(i + 1, (x, y), fontsize=15, color=COLOR_A)
    if not selected_ps:
        plt.grid(linestyle='dashed')
        name = '{}_grid_tsne'.format(DATASET)
    else:
        #plt.axis('off')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(True)
        cur_axes.axes.get_yaxis().set_visible(True)
        name = '{}_{}_tsne'.format(DATASET, conf.name())
    save_fig(plt, DIR, name, print_path=True)
    plt.close()
    
    
def _closest_point(lp, points):
    dists = []
    for point in points:
        dists.append(lp.distance(point))
    id = np.argmin(dists)
    return id, points[id]


def _get_percs(conf):
    return list(np.arange(0, 1, 1 / (conf.num_points - 1))) + [1]


def _get_coord(point):
    return list(point.coords)[0]


if __name__ == '__main__':
    load = False
    save_path_x = join(get_root_path(), DIR, KERNEL, f'binary-updated-dataset-X-{PERF}-bottleneck-tsne-p{perplexity}-n{n_iter}.txt')
    if not load:
        embs, X_pragmas = vis_data()
        with open(save_path_x, "wb") as fp:   #Pickling
            pickle.dump(X_pragmas, fp)
    else:
        with open(save_path_x, "rb") as fp:   #Pickling
            X_pragmas = pickle.load(fp)
        save_path_npy = join(get_root_path(), DIR, KERNEL, f'binary-updated-dataset{PERF}-bottleneck-tsne-p{perplexity}-n{n_iter}.npy')
        embs = np.load(save_path_npy)
        df = pd.read_pickle('test_pickle.pkl')
        df['tsne-2d-one'] = embs[:, 0]
        df['tsne-2d-two'] = embs[:, 1]
        points = _to_points(embs)
        for conf in confs:
            _plot_conf(points, conf, X_pragmas, df['y'])
    
    ######### test np load ########
    # perplexity = 50
    # n_iter = 5000
    # save_path_npy = join(get_root_path(), 'save', f'binary-class-{PERF}-bottleneck-tsne-p{perplexity}-n{n_iter}.npy')
    # tsne_results = np.load(save_path_npy)
    # df = pd.read_pickle('test_pickle.pkl')
    # df['tsne-2d-one'] = tsne_results[:, 0]
    # df['tsne-2d-two'] = tsne_results[:, 1]

    # plt.figure(figsize=(16, 10))
    # sns_plot = sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 1),
    #     data=df,
    #     legend="full",
    #     # alpha=0.3
    # )
    # #sns_plot.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    # save_path = join(get_root_path(), 'save', f'test.png')
    # sns_plot.get_figure().savefig(save_path)
    
