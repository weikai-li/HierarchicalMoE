from config import FLAGS, poly_KERNEL
from utils import MLP, _get_y_with_target, MLP_multi_objective

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, TransformerConv, GCNConv
from torch_geometric.nn import global_add_pool
import torch.nn as nn
from torch_scatter import scatter_add
import random

from nn_att import MyGlobalAttention
from torch.nn import Sequential, Linear, ReLU

from collections import OrderedDict
import copy


class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    k: an integer - how many experts to use for each batch element
    This module is adapted from the code of "Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling"
    """

    def __init__(self, input_size, layer):
        super(MoE, self).__init__()
        self.num_experts = FLAGS.moe_num_experts
        self.k = FLAGS.moe_k
        self.loss_coef = FLAGS.moe_lmbda
        # instantiate experts
        self.experts = nn.ModuleList([copy.deepcopy(layer) for i in range(self.num_experts)])
        self.w_gate = nn.Linear(input_size, self.num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        return x.float().std() / x.float().mean()

    def top_k_gating(self, x):
        """Top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
        """
        logits = self.w_gate(x)

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # print(gates)
        # print(gates.mean(0))
        # exit()
        return gates

    def forward(self, x, edge_index=None, edge_attr=None, data=None, use_ids=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates = self.top_k_gating(x)
        # calculate importance loss
        if use_ids is not None:
            use_gates = gates[use_ids.bool()]
            importance = use_gates.sum(0)
        else:
            importance = gates.mean(0)
        # loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = self.cv_squared(importance)
        loss *= self.loss_coef
        expert_outputs = []
        for i in range(self.num_experts):
            if edge_index is not None:
                expert_i_output = self.experts[i](x, edge_index, edge_attr)
            else:
                expert_i_output = self.experts[i](x)
            expert_outputs.append(expert_i_output)
        if isinstance(expert_i_output, dict):  # This is for output_MLP
            new_output = {}
            for k in expert_i_output.keys():
                y = torch.stack([e[k] for e in expert_outputs], dim=1)
                y = gates.unsqueeze(dim=-1) * y
                y = y.sum(dim=1)
                new_output[k] = y
            y = new_output
        else:
            expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]
            y = gates.unsqueeze(dim=-1) * expert_outputs
            y = y.sum(dim=1)
        if use_ids is not None:
            return (y, loss, use_gates)
        else:
            return (y, loss, gates)


class Net(nn.Module):
    def __init__(self, in_channels, edge_dim=0, init_pragma_dict=None, task=FLAGS.task, 
            num_layers=FLAGS.num_layers, D=FLAGS.D, target=FLAGS.target, no_moe=False,
            moe_layers=FLAGS.moe_layers):
        super(Net, self).__init__()
        
        self.MLP_version = 'multi_obj'  if len(FLAGS.target) > 1 else  'single_obj'
        if FLAGS.gnn_type == 'gat':
            conv_class = GATConv
        elif FLAGS.gnn_type == 'gcn':
            conv_class = GCNConv
        elif FLAGS.gnn_type == 'transformer':
            conv_class = TransformerConv
        else:
            raise NotImplementedError()
        self.moe_layers = moe_layers

        if FLAGS.no_graph:
            if FLAGS.only_pragma:
                self.init_MLPs = nn.ModuleDict()
                for gname, feat_dim in init_pragma_dict.items():
                    mlp = MLP(feat_dim, D,
                                    activation_type=FLAGS.activation,
                                    num_hidden_lyr=1)
                    self.init_MLPs[gname] = mlp
                channels = [D, D, D, D]
                self.conv_first = MLP(D, D,
                                activation_type=FLAGS.activation,
                                hidden_channels=channels,
                                num_hidden_lyr=len(channels))
            else:   
                channels = [D, D, D, D, D]
                self.conv_first = MLP(in_channels, D,
                                activation_type=FLAGS.activation,
                                hidden_channels=channels,
                                num_hidden_lyr=len(channels))
        else:  # This is our case
            if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                # print(in_channels)
                self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
            else:
                self.conv_first = conv_class(in_channels, D)

            self.conv_layers = nn.ModuleList()

            self.num_conv_layers = num_layers - 1
            num_layers += FLAGS.gnn_layer_after_MLP
            for i in range(num_layers - 1):
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    conv = conv_class(D, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
                else:
                    conv = conv_class(D, D)
                if f'gnn{i+2}' in moe_layers and no_moe == False:
                    conv = MoE(D, conv)
                self.conv_layers.append(conv)

            # In our case, gae_T = False, gae_P = False
            if FLAGS.gae_T: # graph auto encoder
                if FLAGS.separate_T:
                    self.gae_transform_T = nn.ModuleDict()
                    for gname, feat_dim in init_pragma_dict.items():
                        mlp = Linear(feat_dim[0], D // 8)
                        if FLAGS.pragma_uniform_encoder:
                            self.gae_transform_T['all'] = Linear(feat_dim[1], D // 8) ## TRY ME: changing the MLP for GAE T, was D // 8
                            break
                        else:
                            self.gae_transform_T[gname] = mlp
                    channels = [D // 2, D // 4]
                    # channels = [D // 2]
                    self.decoder_T = MLP(D, D // 8,
                                activation_type=FLAGS.activation,
                                hidden_channels=channels,
                                num_hidden_lyr=len(channels))
            if FLAGS.gae_P:
                out_channels = in_channels
                if FLAGS.input_encode:
                    self.gate_input = Linear(in_channels, 2 * D) ## encode input one-hot representation
                    out_channels = 2 * D
                
                if FLAGS.decoder_type == 'type1':
                    decoder_arch = []
                elif FLAGS.decoder_type == 'type2':
                    decoder_arch = [D, 2 * D, out_channels]
                self.decoder_P = MLP(D, out_channels, activation_type = FLAGS.activation,
                                hidden_channels = decoder_arch,
                                num_hidden_lyr = len(decoder_arch))
                if FLAGS.decoder_type == 'None':
                    for name, param in self.decoder_P.named_parameters():
                        print(name)
                        param.requires_grad = False
            if FLAGS.gae_T or FLAGS.gae_P:
                self.gae_sim_function = nn.CosineSimilarity()
                self.gae_loss_function = nn.CosineEmbeddingLoss()

        self.jkn = JumpingKnowledge(FLAGS.jkn_mode, channels=D, num_layers=2)

        self.task = task
        self.no_moe = no_moe

        if task == 'regression':
            if 'GNLL' in FLAGS.loss:
                self.out_dim = 1
                self.MLP_out_dim = 2
                if FLAGS.loss == 'myGNLL':
                    self.loss_function = self.gaussianNLL
                else:
                    self.loss_function = nn.GaussianNLLLoss()
                    self.my_softplus = nn.Softplus()
                
            else:
                self.out_dim = 1
                self.MLP_out_dim = 1
                self.loss_function = nn.MSELoss()
        else:
            self.out_dim = 2
            self.MLP_out_dim = 2
            self.loss_function = nn.CrossEntropyLoss()

        
        
        if FLAGS.node_attention:
            if FLAGS.separate_T:
                self.gate_nn_T = self.node_att_gate_nn(D)
                self.glob_T = MyGlobalAttention(self.gate_nn_T, None)
            self.separate_P = FLAGS.separate_P
            if FLAGS.separate_pseudo: ## for now, only pseudo node for block
                self.gate_nn_pseudo_B = self.node_att_gate_nn(D)
                if len(moe_layers) > 0 and moe_layers[0][:6] == 'pseudo' and no_moe == False:
                    self.pseudo_moe = MoE(D, nn.Linear(D, D))
                    if moe_layers[0][:12] == 'pseudo_alone':
                        self.separate_P = False
                self.glob_pseudo_B = MyGlobalAttention(self.gate_nn_pseudo_B, None)
            if self.separate_P:
                self.gate_nn_P = self.node_att_gate_nn(D)
                self.glob_P = MyGlobalAttention(self.gate_nn_P, None)
            if FLAGS.separate_icmp:
                self.gate_nn_icmp = self.node_att_gate_nn(D)
                self.glob_icmp = MyGlobalAttention(self.gate_nn_icmp, None)

        
        if 'regression' in self.task:
            _target_list = target
            if not isinstance(FLAGS.target, list):
                _target_list = [target]
            # if FLAGS.new_speedup == False:
            #     self.target_list = [t for t in _target_list if t != 'perf' else 'kernel_speedup'] # to use with trained model from old speedup
            # else
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        
        if not FLAGS.SSL:
            if FLAGS.node_attention:
                dim = FLAGS.separate_T + self.separate_P + FLAGS.separate_pseudo + FLAGS.separate_icmp
                in_D = dim * D
            else:
                in_D = D
            if D > 64:
                hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
            else:
                hidden_channels = [D // 2, D // 4, D // 8]

            assert self.MLP_version == 'multi_obj'
            if self.MLP_version == 'single_obj':
                self.MLPs = nn.ModuleDict()
                for target in self.target_list:
                    self.MLPs[target] = MLP(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                            hidden_channels=hidden_channels,
                                            num_hidden_lyr=len(hidden_channels))
            else:  # this is our case
                self.MLPs = MLP_multi_objective(in_D, self.MLP_out_dim, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels,
                                        objectives=self.target_list,
                                        num_common_lyr=FLAGS.MLP_common_lyr)
                if 'output_mlp' in moe_layers and no_moe == False:
                    self.MLPs = MoE(in_D, self.MLPs)
                
        ## pragma as MLP
        if FLAGS.pragma_as_MLP:
            self.pragma_as_MLP_list = FLAGS.pragma_as_MLP_list
            self.MLPs_per_pragma = nn.ModuleDict()
            for target in self.pragma_as_MLP_list:   # In our case, it has tile, pipeline, parallel
                in_D = D + 1
                if target == 'parallel': in_D = D + 2 ## reduction/normal, factor
                hidden_channels, len_hidden_channels = None, 0
                if FLAGS.pragma_MLP_hidden_channels is not None:
                    hidden_channels = eval(FLAGS.pragma_MLP_hidden_channels)
                    len_hidden_channels = len(hidden_channels)
                self.MLPs_per_pragma[target] = MLP(in_D, D, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels, num_hidden_lyr=len_hidden_channels)
                if "pragma_mlp" in moe_layers and no_moe == False:
                    self.MLPs_per_pragma[target] = MoE(in_D, self.MLPs_per_pragma[target])
            if FLAGS.pragma_order == 'parallel_and_merge':   # In our case, it is true
                in_D = D * len(self.pragma_as_MLP_list)
                hidden_channels = eval(FLAGS.merge_MLP_hidden_channels)
                
                self.MLPs_per_pragma['merge'] = MLP(in_D, D, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels, num_hidden_lyr=len(hidden_channels))
                if "pragma_merge" in moe_layers and no_moe == False:
                    self.MLPs_per_pragma["merge"] = MoE(in_D, self.MLPs_per_pragma["merge"])

    def node_att_gate_nn(self, D):
        assert FLAGS.node_attention_MLP == False
        if FLAGS.node_attention_MLP:
            return MLP(D, 1,
                    activation_type=FLAGS.activation_type,
                    hidden_channels=[D // 2, D // 4, D // 8],
                    num_hidden_lyr=3)
        else:   # This is our case
            if ('pooling' in self.moe_layers or 'pseudo_alone+pooling' in self.moe_layers) and self.no_moe == False:
                return MoE(D, Sequential(Linear(D, D), ReLU(), Linear(D, 1)))
            else:
                return Sequential(Linear(D, D), ReLU(), Linear(D, 1))

    def cal_gae_loss(self, encoded_g, decoded_out):
        target = torch.ones(len(encoded_g), device=FLAGS.device)  ## for similarity, use the negative form for dissimilarity
        target.requires_grad = False
        gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        return gae_loss
    
    def gaussianNLL(self, out=None, target=None):
        '''
            out should include mu and sigma
            https://towardsdatascience.com/get-uncertainty-estimates-in-neural-networks-for-free-48f2edb82c8f
        '''
        if out is not None:
            mu = out[:, 0].reshape(-1, 1)
            log_var = out[:, 1].reshape(-1, 1)
            # mu = torch.zeros(log_var.shape).to(FLAGS.device)
            var = torch.exp(log_var)
            
            if FLAGS.beta > 0:
                scalar = (var.detach() ** FLAGS.beta)
            else:
                scalar = torch.ones(var.shape).to(FLAGS.device)
        

            return torch.mean((log_var / 2 + (1/2) * (1/var) * ((target - mu))**2) * scalar) 
        else:
            return None
        
    def mask_emb(self, out, non_zero_ids):
        out = out.permute((1, 0))
        out = out * non_zero_ids
        out = out.permute((1, 0))
        
        return out
    
    
    def apply_pragam_as_MLP(self, mlp_pragma, out, scope_nodes, X_pragma_per_node, ptype):
        if ptype == 'tile':
            pragma_option = X_pragma_per_node[:, 0].reshape(-1, 1)
        elif ptype == 'pipeline':
            pragma_option = X_pragma_per_node[:, 1].reshape(-1, 1)
        elif ptype == 'parallel':
            pragma_option = X_pragma_per_node[:, 2:4].reshape(-1, 2)
        elif ptype == 'merge':
            mlp_inp = X_pragma_per_node
        else:
            raise NotImplementedError()
            
        non_scope_nodes = torch.sub(1, scope_nodes)
        masked_emb = scope_nodes.ge(0.5)
        if ptype == 'merge':
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            if "pragma_merge" in self.moe_layers and self.no_moe == False:
                mlp_out, loss, gates = mlp_out
            else:
                loss = None
            out = self.mask_emb(out.clone(), non_zero_ids=non_scope_nodes) + self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
        else:
            mlp_inp = torch.cat((out, pragma_option), dim=1)
            # mlp_out = mlp_pragma(mlp_inp[masked_emb])
            # out = torch.clone(out)
            # out[masked_emb] = mlp_out
            mlp_out = mlp_pragma(self.mask_emb(mlp_inp, non_zero_ids=scope_nodes))
            if "pragma_mlp" in self.moe_layers and self.no_moe == False:
                mlp_out, loss, gates = mlp_out
            else:
                loss = None
            if FLAGS.pragma_order == 'sequential':
                out = self.mask_emb(out.clone(), non_zero_ids=non_scope_nodes) + self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
            elif FLAGS.pragma_order == 'parallel_and_merge':
                out = self.mask_emb(mlp_out, non_zero_ids=scope_nodes)
            else:
                raise NotImplementedError()
        
        if loss is None:
            return out
        else:
            return out, loss, gates
    
    # TODO: return_middle = i means to return the hidden representation after i GNNs; 0 means no return
    def forward(self, data, return_middle=0,
            return_gate='', test_mode=False, return_extra_loss=False, **kwargs):
        x, edge_index, edge_attr, batch, pragmas = \
            data.x, data.edge_index, data.edge_attr, data.batch, data.pragmas

        if hasattr(data, 'kernel'):
            gname = data.kernel[0]
        if hasattr(data, 'X_pragma_per_node'):
            X_pragma_per_node = data.X_pragma_per_node
        # print(gname)
        # print(edge_attr.shape)
        outs = []
        out_dict = OrderedDict()
        if FLAGS.activation == 'relu':
            activation = F.relu
        elif FLAGS.activation == 'elu':
            activation = F.elu
        else:
            raise NotImplementedError()
        
        extra_loss = 0
        if FLAGS.no_graph:
            out = x
            if FLAGS.only_pragma:
                MLP_to_use = self.init_MLPs[gname]
                out = MLP_to_use(pragmas)
            
            out = self.conv_first(out)
        else:
            if FLAGS.encode_edge and  FLAGS.gnn_type == 'transformer':
                out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
            else:
                out = activation(self.conv_first(x, edge_index))

            outs.append(out)

            # for i, conv in enumerate(self.conv_layers):
            assert self.num_conv_layers == 5
            for i in range(self.num_conv_layers):
                if return_middle == i + 1:
                    return out.detach().clone()
                conv = self.conv_layers[i]
                if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                    if f'gnn{i+2}' in self.moe_layers and self.no_moe == False:
                        out, loss, gates = conv(out, edge_index, edge_attr=edge_attr, data=data)
                        if return_gate == f'gnn{i+2}':
                            return gates
                        extra_loss += loss
                    else:
                        out = conv(out, edge_index, edge_attr=edge_attr)
                else:
                    out = conv(out, edge_index)
                if i != len(self.conv_layers) - 1:
                    out = activation(out)
                    
                outs.append(out)

            # TODO: this is our case
            if FLAGS.jkn_enable:
                out = self.jkn(outs)
        
        if return_middle == 6:
            copy_hidden = out.detach().clone()
            size = batch[-1].item() + 1
            num_node = (batch == 0).sum()
            for i in range(size):
                try:
                    assert num_node == (batch == i).sum()
                except:
                    assert poly_KERNEL[0] == '2mm' and len(poly_KERNEL) == 1
            copy_hidden = scatter_add(copy_hidden, batch, dim=0, dim_size=size)
            copy_hidden = copy_hidden / num_node  # In this way, the scale is the same to every kernel
            return copy_hidden

        ## pragma as MLP
        if FLAGS.pragma_as_MLP:
            in_merge = None
            for pragma in self.pragma_as_MLP_list:
                out_MLP = self.apply_pragam_as_MLP(self.MLPs_per_pragma[pragma], out, \
                                        data.X_pragmascopenids, X_pragma_per_node, pragma)
                if "pragma_mlp" in self.moe_layers and self.no_moe == False:
                    out_MLP, loss, gates = out_MLP
                    if return_gate == 'pragma_mlp':
                        return gates
                    extra_loss += loss
                if FLAGS.pragma_order == 'sequential':
                    out = out_MLP
                elif FLAGS.pragma_order == 'parallel_and_merge':
                    if in_merge is None: in_merge = out_MLP
                    else: in_merge = torch.cat((in_merge, out_MLP), dim=1)
                else:
                    raise NotImplementedError()
            ## the merge part
            if FLAGS.pragma_order == 'parallel_and_merge':
                out = self.apply_pragam_as_MLP(self.MLPs_per_pragma['merge'], out, \
                                         data.X_pragmascopenids, in_merge, 'merge')
                if "pragma_merge" in self.moe_layers and self.no_moe == False:
                    out, loss, gates = out
                    if return_gate == 'pragma_merge':
                        return gates
                    extra_loss += loss
            
            if return_middle == 6.3:   # Before gnn7 and after pragma mlp
                return out.detach().clone()
            
            if 'pseudo_alone_w/o_gnn7' not in self.moe_layers or self.no_moe == True:
                for i, conv in enumerate(self.conv_layers[self.num_conv_layers:]):
                    if FLAGS.encode_edge and FLAGS.gnn_type == 'transformer':
                        if 'gnn7' in self.moe_layers and self.no_moe == False:
                            out, loss, gates = conv(out, edge_index, edge_attr=edge_attr, data=data)
                            if return_gate == 'gnn7':
                                return gates
                            extra_loss += loss
                        else:
                            out = conv(out, edge_index, edge_attr=edge_attr)
                    else:
                        out = conv(out, edge_index)
                    if i != len(self.conv_layers) - 1:
                        out = activation(out)

        if return_middle == 6.5:
            return out[data['X_pseudonids'].bool()].detach().clone()

        if FLAGS.node_attention:
            out_gnn = out
            out_g = None
            out_P, out_T = None, None
            if self.separate_P:
                if FLAGS.P_use_all_nodes:
                    if ('pooling' in self.moe_layers or 'pseudo_alone+pooling' in self.moe_layers) and self.no_moe == False:
                        out_P, node_att_scores_P, loss = self.glob_P(out_gnn, batch)
                        extra_loss += loss
                    else:
                        out_P, node_att_scores_P = self.glob_P(out_gnn, batch)
                else:
                    if ('pooling' in self.moe_layers or 'pseudo_alone+pooling' in self.moe_layers) and self.no_moe == False:
                        out_P, node_att_scores_P, loss = self.glob_P(out_gnn, batch, set_zeros_ids=data.X_contextnids)
                        extra_loss += loss
                    else:
                        out_P, node_att_scores_P = self.glob_P(out_gnn, batch, set_zeros_ids=data.X_contextnids)
                out_dict['emb_P'] = out_P
                out_g = out_P
                
            if FLAGS.separate_T:
                out_T, node_att_scores = self.glob_T(out_gnn, batch, set_zeros_ids=data.X_pragmanids)
                out_dict['emb_T'] = out_T
                if out_P is not None:
                    out_g = torch.cat((out_P, out_T), dim=1)
                else:
                    out_g = out_T
                    
            if FLAGS.separate_pseudo:
                if len(self.moe_layers) > 0 and self.moe_layers[0][:6] == 'pseudo' and self.no_moe == False:
                    out_gnn, loss, gates = self.pseudo_moe(out_gnn, use_ids=data.X_pseudonids)
                    if return_gate == 'pseudo_alone_w/o_gnn7' and self.moe_layers[0] == 'pseudo_alone_w/o_gnn7':
                        return gates
                    extra_loss += loss
                if ('pooling' in self.moe_layers or 'pseudo_alone+pooling' in self.moe_layers) and self.no_moe == False:
                    out_pseudo_B, node_att_scores_pseudo, loss = self.glob_pseudo_B(out_gnn, batch, set_zeros_ids=data.X_pseudonids)
                    extra_loss += loss
                else:
                    out_pseudo_B, node_att_scores_pseudo = self.glob_pseudo_B(out_gnn, batch, set_zeros_ids=data.X_pseudonids)
                out_dict['emb_pseudo_b'] = out_pseudo_B
                # if out_pseudo_B.absolute().sum() > 0:    # If we use the GNN-DSE graph, it will be zero.
                if out_g is not None:
                    out_g = torch.cat((out_g, out_pseudo_B), dim=1)
                else:
                    out_g = out_pseudo_B

            if FLAGS.separate_icmp:
                out_icmp, node_att_scores_icmp = self.glob_icmp(out_gnn, batch, set_zeros_ids=data.X_icmpnids)
                out_dict['emb_icmp'] = out_icmp
                if out_g is not None:
                    out_g = torch.cat((out_g, out_icmp), dim=1)
                else:
                    out_g = out_icmp             
            
            if not self.separate_P and not FLAGS.separate_T and not FLAGS.separate_pseudo:
                out_g, node_att_scores = self.glob_T(out_gnn, batch)
                out_dict['emb_T'] = out
                if FLAGS.subtask == 'visualize':
                    from saver import saver
                    saver.save_dict({'data': data, 'node_att_scores': node_att_scores},
                                    f'node_att.pickle')
                    
            out = out_g
            if return_middle == 7:
                return out.detach().clone()
        else:
            out = global_add_pool(out, batch)
            out_dict['emb_T'] = out

        total_loss = 0
        if test_mode == False:
            total_loss += extra_loss
        
        gae_loss = 0
        # In our case, FLAGS.gae_P = False, FLAGS.gae_T = False
        if FLAGS.gae_T: # graph auto encoder
            assert FLAGS.separate_T
            if FLAGS.pragma_uniform_encoder:
                gname = 'all'
            encoded_g = self.gae_transform_T[gname](pragmas)
            decoded_out = self.decoder_T(out_dict['emb_T'])
            gae_loss = self.cal_gae_loss(encoded_g, decoded_out)
            # target = torch.ones(len(encoded_g), device=FLAGS.device) ## for similarity, use the negative form for dissimilarity
            # target.requires_grad = False
            # gae_loss = self.gae_loss_function(encoded_g, decoded_out, target)
        if FLAGS.gae_P:
            assert self.separate_P
            encoded_x = x
            if FLAGS.input_encode:
                encoded_x = self.gate_input(x)
            encoded_g = global_add_pool(encoded_x, batch) ## simple addition of node embeddings for gae
            
            if FLAGS.decoder_type == 'None': ## turn off autograd:
                decoded_out = self.decoder_P(out_dict['emb_P']).detach()
            else: 
                decoded_out = self.decoder_P(out_dict['emb_P']).to(FLAGS.device)
            # gae_loss = (self.gae_loss_function(encoded_g, decoded_out)).mean()
            gae_loss += self.cal_gae_loss(encoded_g, decoded_out)
            # from saver import saver
            # saver.info(f'cosine similarity is {self.gae_sim_function(encoded_g, decoded_out).mean()}')
            # saver.log_info(f'encoded_g : {F.normalize(encoded_g[0, :], dim=0)}')
            # saver.log_info(f'decoded_out : {F.normalize(decoded_out[0, :], dim=0)}')
        if FLAGS.gae_P or FLAGS.gae_T:
            total_loss += torch.abs(gae_loss)
                # gae_loss = gae_loss.view((len(gae_loss), 1))
                # print(gae_loss.shape)

        # out, edge_index, _, batch, perm, score = self.pool1(
        #     out, edge_index, None, batch)
        # ratio = out.size(0) / x.size(0)            

        # loss_dict = OrderedDict()
        
        out_embed = out
        # print(out.shape)
        # exit()
        # _target_list = FLAGS.target
        # if not isinstance(FLAGS.target, list):
        #     _target_list = [FLAGS.target]
        # target_list = [t for t in _target_list]
        # target_list = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
        # target_list = ['util-DSP']
        
        loss_dict = {}
        if not FLAGS.SSL:
            assert self.MLP_version == 'multi_obj'
            if self.MLP_version == 'multi_obj':
                if "output_mlp" in self.moe_layers and self.no_moe == False:
                    out_MLPs, loss, gates = self.MLPs(out_embed, data=data)
                    if return_gate == 'output_mlp':
                        return gates
                    if test_mode == False:
                        total_loss += loss
                        extra_loss += loss
                else:
                    out_MLPs = self.MLPs(out_embed)

            for target_name in self.target_list:
            # for target_name in target_list:
                if self.MLP_version == 'multi_obj':
                    out = out_MLPs[target_name]
                else:
                    out = self.MLPs[target_name](out_embed)
                y = _get_y_with_target(data, target_name)
                if self.task == 'regression':
                    target = y.view((len(y), self.out_dim))
                    assert target.shape == out.shape
                    # print('target', target.shape)
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_function(out, target))
                        # loss = mean_squared_error(target, out, squared=False)
                    elif FLAGS.loss == 'MSE' or FLAGS.loss == 'myGNLL':
                        loss = self.loss_function(out, target) ## predicting my and log_var in case of myGNLL
                        # loss = mean_squared_error(target, out, squared=True)
                    elif FLAGS.loss == 'GNLL': ## predicting var
                        var = self.my_softplus(out[:, 1].reshape(-1, 1))
                        loss = self.loss_function(out[:, 0].reshape(-1, 1), target, var)
                    else:
                        raise NotImplementedError()
                    # print('loss', loss.shape)
                else:
                    target = y.view((len(y)))
                    # out = out.argmax(1)
                    # print(out.shape, target.shape)
                    # acc = (out == target).sum() / len(out)
                    # print(acc)
                    # exit()
                    loss = self.loss_function(out, target)
                out_dict[target_name] = out
                total_loss += loss
                loss_dict[target_name] = loss
            
            if return_middle == 8:
                return out_dict

        if return_extra_loss:
            return out_dict, total_loss, loss_dict, gae_loss, extra_loss
        else:
            return out_dict, total_loss, loss_dict, gae_loss


    # TODO: only calculate the total_loss (I deleted a lot of things, I need to add them back)
    def calculate_total_loss(self, data, out_dict, return_loss_dict=False):
        total_loss = 0
        loss_dict = {}
        assert not FLAGS.gae_P and not FLAGS.gae_T
        if not FLAGS.SSL:
            for target_name in self.target_list:
                y = _get_y_with_target(data, target_name)
                assert self.task == 'regression'
                target = y.view((len(y), self.out_dim))
                out = out_dict[target_name]
                assert target.shape == out.shape
                if FLAGS.loss == 'RMSE':
                    loss = torch.sqrt(self.loss_function(out, target))
                elif FLAGS.loss == 'MSE' or FLAGS.loss == 'myGNLL':
                    loss = self.loss_function(out, target)
                elif FLAGS.loss == 'GNLL': ## predicting var
                    var = self.my_softplus(out[:, 1].reshape(-1, 1))
                    loss = self.loss_function(out[:, 0].reshape(-1, 1), target, var)
                else:
                    raise NotImplementedError()
                loss_dict[target_name] = loss
                total_loss += loss
        if return_loss_dict:
            return total_loss, loss_dict
        else:
            return total_loss


class HierarchicalMoE(nn.Module):
    def __init__(self, in_channels, edge_dim = 0, init_pragma_dict = None, task = FLAGS.task, 
            num_layers = FLAGS.num_layers, D = FLAGS.D, target = FLAGS.target, no_moe = False):
        super(HierarchicalMoE, self).__init__()
        self.loss_coef = FLAGS.hierarchical_moe_lmbda
        # instantiate experts
        assert len(FLAGS.moe_layers) == 1
        self.moe_layers = FLAGS.moe_layers[0]

        self.conv_first = TransformerConv(in_channels, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
        
        if FLAGS.hierarchical_moe_component == None:
            self.expert_names = ['gnn7', 'pseudo_alone_w/o_gnn7', 'output_mlp']
            moe1 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['gnn7'])
            moe2 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['pseudo_alone_w/o_gnn7'])
            moe3 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['output_mlp'])
            total_hidden = 5 * D
        elif FLAGS.hierarchical_moe_component == 'gnn7':
            self.expert_names = ['gnn7', 'gnn7', 'gnn7']
            moe1 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['gnn7'])
            moe2 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['gnn7'])
            moe3 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['gnn7'])
            total_hidden = 6 * D
        elif FLAGS.hierarchical_moe_component == 'pseudo_alone_w/o_gnn7':
            self.expert_names = ['pseudo_alone_w/o_gnn7', 'pseudo_alone_w/o_gnn7', 'pseudo_alone_w/o_gnn7']
            moe1 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['pseudo_alone_w/o_gnn7'])
            moe2 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['pseudo_alone_w/o_gnn7'])
            moe3 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['pseudo_alone_w/o_gnn7'])
            total_hidden = 3 * D
        elif FLAGS.hierarchical_moe_component == 'output_mlp':
            self.expert_names = ['output_mlp', 'output_mlp', 'output_mlp']
            moe1 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['output_mlp'])
            moe2 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['output_mlp'])
            moe3 = Net(in_channels, edge_dim, init_pragma_dict, task, num_layers, D, target,
                        no_moe, moe_layers=['output_mlp'])
            total_hidden = 6 * D
        if FLAGS.separate_pseudo == False:
            total_hidden = 3 * D
        
        if FLAGS.hierarchical_share_layers:
            moe1.conv_first = self.conv_first
            moe2.conv_first = self.conv_first
            moe3.conv_first = self.conv_first

            for i in range(5):
                conv = TransformerConv(D, D, edge_dim=edge_dim, dropout=FLAGS.dropout)
                moe1.conv_layers[i] = conv
                moe2.conv_layers[i] = conv
                moe3.conv_layers[i] = conv

        self.experts = nn.ModuleList([moe1, moe2, moe3])
        
        self.num_experts = 3
        if FLAGS.moe_layers[0] in ['hierarchy-weighted-input', 'hierarchy-weighted-hidden']:
            self.top_k = 3
        else:
            self.top_k = 1

        self.MLP_version = 'multi_obj' if len(FLAGS.target) > 1 else 'single_obj'
        if FLAGS.moe_layers[0] in ['hierarchy-weighted-input', 'hierarchy-top-input']:
            self.gate_nn = self.node_att_gate_nn(in_channels, D)
            self.glob_gate = MyGlobalAttention(self.gate_nn, None)
            self.gate_linear = nn.Linear(in_channels, self.num_experts)
        else:
            assert FLAGS.moe_layers[0] in ['hierarchy-weighted-hidden', 'hierarchy-top-hidden']
            self.gate_linear = nn.Linear(total_hidden, self.num_experts)
        if FLAGS.hierarchical_moe_component == None:
            nn.init.constant_(self.gate_linear.weight, self.gate_linear.weight.data.mean())
            nn.init.constant_(self.gate_linear.bias, 0)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.task = task
        if 'regression' in task:
            _target_list = target
            if not isinstance(FLAGS.target, list):
                _target_list = [target]
            # if FLAGS.new_speedup == False:
            #     self.target_list = [t for t in _target_list if t != 'perf' else 'kernel_speedup'] # to use with trained model from old speedup
            # else
            self.target_list = [t for t in _target_list]
        else:
            self.target_list = ['perf']
        if task == 'regression':
            if 'GNLL' in FLAGS.loss:
                self.out_dim = 1
                self.MLP_out_dim = 2
                if FLAGS.loss == 'myGNLL':
                    self.loss_function = self.gaussianNLL
                else:
                    self.loss_function = nn.GaussianNLLLoss()
                    self.my_softplus = nn.Softplus()
            else:
                self.out_dim = 1
                self.MLP_out_dim = 1
                self.loss_function = nn.MSELoss()
        else:
            self.out_dim = 2
            self.MLP_out_dim = 2
            self.loss_function = nn.CrossEntropyLoss()

    def node_att_gate_nn(self, in_channels, D):
        assert FLAGS.node_attention_MLP == False
        if FLAGS.node_attention_MLP:
            pass
        else:   # This is our case
            return Sequential(Linear(in_channels, D), ReLU(), Linear(D, 1))
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        return x.float().std() / x.float().mean()

    def top_k_gating(self, x, batch=None):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
        """
        if FLAGS.moe_layers[0] in ['hierarchy-weighted-input', 'hierarchy-top-input']:
            batched_emb, node_att_scores = self.glob_gate(x, batch)
            logits = self.gate_linear(batched_emb)
        else:
            logits = self.gate_linear(x)
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(self.num_experts, dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        return gates

    def calculate_total_loss(self, data, out_dict):
        total_loss = 0
        loss_dict = {}
        if not FLAGS.SSL:
            assert self.MLP_version == 'multi_obj'
            if self.MLP_version == 'multi_obj':
                pass
            for target_name in self.target_list:
                if self.MLP_version == 'multi_obj':
                    out = out_dict[target_name]
                y = _get_y_with_target(data, target_name)
                if self.task == 'regression':
                    target = y.view((len(y), self.out_dim))
                    assert target.shape == out.shape
                    if FLAGS.loss == 'RMSE':
                        loss = torch.sqrt(self.loss_function(out, target))
                        # loss = mean_squared_error(target, out, squared=False)
                    elif FLAGS.loss == 'MSE' or FLAGS.loss == 'myGNLL':
                        loss = self.loss_function(out, target) ## predicting my and log_var in case of myGNLL
                        # loss = mean_squared_error(target, out, squared=True)
                    elif FLAGS.loss == 'GNLL': ## predicting var
                        var = self.my_softplus(out[:, 1].reshape(-1, 1))
                        loss = self.loss_function(out[:, 0].reshape(-1, 1), target, var)
                    else:
                        raise NotImplementedError()
                else:
                    target = y.view((len(y)))
                    loss = self.loss_function(out, target)
                total_loss += loss
                loss_dict[target_name] = loss
        return total_loss, loss_dict

    def forward(self, data, return_middle=0, return_gate='', test_mode=False, epoch=None):
        outputs = []
        hiddens = []
        total_extra_loss = 0
        return_middle_dict = {6.3: 'gnn7', 6.5: 'pseudo_alone_w/o_gnn7', 7: 'output_mlp', 9: 'hierarchy-weighted-hidden'}

        for i in range(self.num_experts):
            if return_gate in self.expert_names:
                if return_gate == self.expert_names[i]:
                    gates = self.experts[i](data, return_middle, return_gate, test_mode)
                    return gates
            elif return_middle != 0 and return_middle != 8:
                if return_middle_dict[return_middle] in self.expert_names:
                    if self.expert_names[i] == return_middle_dict[return_middle]:
                        hidden = self.experts[i](data, return_middle, return_gate, test_mode)
                        return hidden
                    else:
                        continue
                else:
                    hidden = self.experts[i](data, 7, return_gate, test_mode)
                    hiddens.append(hidden)
            else:
                out_dict, total_loss, loss_dict, gae_loss, extra_loss = self.experts[i](data,
                    0, return_gate, test_mode, return_extra_loss=True)
                outputs.append(out_dict)
                total_extra_loss += extra_loss
                hidden = self.experts[i](data, 7, True, return_gate, test_mode)
                hiddens.append(hidden)
        total_extra_loss = total_extra_loss / self.num_experts

        separate_train = False
        if epoch is not None and FLAGS.finetune == False:
            if epoch < FLAGS.hierarchical_moe_epoch or \
                    (FLAGS.hierarchical_alternate_train == True and epoch % 2 == 1):
                separate_train = True

        if separate_train:
            # In the first a few epochs, train the three expert networks separately
            total_loss1, loss_dict1 = self.calculate_total_loss(data, outputs[0])
            total_loss2, loss_dict2 = self.calculate_total_loss(data, outputs[1])
            total_loss3, loss_dict3 = self.calculate_total_loss(data, outputs[2])
            total_loss = total_loss1 + total_loss2 + total_loss3
            total_loss = total_loss / 3
            if test_mode == False:
                total_loss += total_extra_loss
            return outputs[0], total_loss, loss_dict1, 0

        else:
            if FLAGS.moe_layers[0] in ['hierarchy-weighted-input', 'hierarchy-top-input']:
                gates = self.top_k_gating(data.x, data.batch)
            else:
                hiddens = torch.concat(hiddens, 1)
                gates = self.top_k_gating(hiddens)
            if return_middle != 0 and return_middle != 8:
                assert FLAGS.moe_layers[0][0:9] == 'hierarchy'
                if return_middle == 0.5:
                    raise NotImplementedError()
                return hiddens
            if return_gate[0:9] == 'hierarchy':
                return gates

            importance = gates.sum(0)
            moe_loss = self.cv_squared(importance)
            moe_loss *= self.loss_coef
            total_extra_loss += moe_loss

            # Aggregate the outputs
            new_outputs = {}
            for k in self.target_list:
                y = torch.stack([e[k] for e in outputs], dim=1)
                y = gates.unsqueeze(dim=-1) * y
                y = y.sum(dim=1)
                new_outputs[k] = y
            
            if return_middle == 8:
                return new_outputs

            total_loss, loss_dict = self.calculate_total_loss(data, new_outputs)
            if test_mode == False:
                total_loss += total_extra_loss
            return new_outputs, total_loss, loss_dict, 0
