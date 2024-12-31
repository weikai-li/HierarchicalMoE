import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from sklearn.cluster import KMeans

import config
poly_KERNEL = config.poly_KERNEL
from config import FLAGS
from model import Net, HierarchicalMoE


def get_hidden(li, pragma_dim):
    from data import MyOwnDataset
    whole_dataset = MyOwnDataset(data_files=li)
    loader = DataLoader(whole_dataset, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False, num_workers=0)
    feat_list = []
    num_features = loader.dataset[0].num_features
    edge_dim = loader.dataset[0].edge_attr.shape[1]
    if len(FLAGS.moe_layers) > 0 and FLAGS.moe_layers[0][0:9] == 'hierarchy':
        assert len(FLAGS.moe_layers) == 1
        model = HierarchicalMoE(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    else:
        model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
    model_path = FLAGS.model_path[0] if type(FLAGS.model_path) is list else FLAGS.model_path 
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    feat_list = []

    for data in loader:
        data = data.to(FLAGS.device)
        hidden_rep = model.to(FLAGS.device)(data, return_middle=7)
        feat_list.extend(hidden_rep.reshape(len(data), -1))
    
    del model
    feat_list = torch.stack(feat_list)
    return feat_list


def kmeans_split(file_li, lengths, pragma_dim):
    feat_list = get_hidden(file_li, pragma_dim)
    train_len, val_len, test_len = lengths[0], lengths[1], lengths[2]
    assert val_len == 0
    feat_list = [f.cpu().numpy() for f in feat_list]
    kmeans = KMeans(n_clusters=train_len, max_iter=300).fit(feat_list)
    centroids, labels = kmeans.cluster_centers_, kmeans.labels_
    mn = np.ones((train_len), dtype=np.float32) * 10e6
    selected_indices = np.ones((train_len), dtype=np.int32) * -1
    for i in range(len(feat_list)):
        centroid = centroids[labels[i]]
        dis = np.linalg.norm(feat_list[i] - centroid, ord=2)
        assert dis < 10e5
        if dis < mn[labels[i]]:
            mn[labels[i]] = dis
            selected_indices[labels[i]] = i
    
    selected_indices = list(set(selected_indices))
    if -1 in selected_indices:
        selected_indices.remove(-1)
    train_li = [file_li[i] for i in selected_indices]
    unselected_li = []
    for i, li in enumerate(file_li):
        if i not in selected_indices:
            unselected_li.append(li)
    while len(train_li) < train_len:
        train_li.append(unselected_li[0])
        unselected_li = unselected_li[1:]
    li = random_split(unselected_li, [val_len, test_len], generator=torch.Generator().manual_seed(FLAGS.random_seed))
    return [train_li, li[0], li[1]], selected_indices
