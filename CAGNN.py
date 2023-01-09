import datetime
import math

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch as th
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from tqdm import tqdm
import torch.nn.functional as F

from graph.aggregator import LocalAggregator


# from skip_edge_gnn import HeteroGraphConv


class CAGNN(nn.Module):

    def __init__(self,
                 args,
                 num_item,
                 num_cat,
                 batch_norm=True,
                 feat_drop=0.0,
                 attention_drop=0.0):
        super(CAGNN, self).__init__()

        self.alpha = args.alpha
        self.embedding_dim = args.emb_size
        self.graph_feature_select = args.graph_feature_select
        self.aux_factor = 2  # hyper-parameter for aux information size
        # self.auxemb_dim = int(self.embedding_dim // self.aux_factor)
        self.item_embedding = nn.Embedding(num_item, self.embedding_dim, max_norm=1)
        self.cate_embedding = nn.Embedding(num_cat, self.embedding_dim, max_norm=1)
        self.pos_embedding = nn.Embedding(200, self.embedding_dim)

        self.num_layers = args.num_layers  # hyper-parameter for gnn layers
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim * 2) if batch_norm else None

        self.finalfeature = FeatureSelect(self.embedding_dim, type=self.graph_feature_select)
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.gnn_layers.append(
                dglnn.HeteroGraphConv({
                    'i2i':
                        GATConv(self.embedding_dim,
                                self.embedding_dim,
                                feat_drop=feat_drop,
                                attn_drop=attention_drop),
                    'c2c':
                        GATConv(self.embedding_dim,
                                self.embedding_dim,
                                feat_drop=feat_drop,
                                attn_drop=attention_drop),
                    'c2i':
                        GATConv(self.embedding_dim,
                                self.embedding_dim,
                                feat_drop=feat_drop,
                                attn_drop=attention_drop),
                    'i2c':
                        GATConv(self.embedding_dim,
                                self.embedding_dim,
                                feat_drop=feat_drop,
                                attn_drop=attention_drop),
                }, aggregate='sum'))

        # W_h_e * (h_s || e_u) + b
        # self.W_pos = nn.Parameter(th.Tensor(self.embedding_dim * 2 + self.auxemb_dim, self.embedding_dim))
        self.W_pos = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.W_hs_e = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.W_h_e = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.W_c = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=False)
        self.local_agg = LocalAggregator(self.embedding_dim, self.alpha, dropout=0.0)
        self.dropout_local = args.dropout_local
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.embedding_dim, self.embedding_dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.embedding_dim, 1))
        self.glu1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.glu2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.reset_parameters()




    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def handle_local(self, hidden, mask):

        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        # scores = torch.matmul(select, b.transpose(1, 0))
        return select

    def feature_encoder(self, g: dgl.DGLHeteroGraph):
        iid = g.nodes['i'].data['id']
        cid = g.nodes['c'].data['id']

        # store the embedding in graph
        # g.update_all(fn.copy_e('pos', 'ft'),
        #              fn.min('ft', 'f_pos'),
        #              etype='c2i')
        # pos_emb = self.pos_embedding(g.nodes['i'].data['f_pos'].long())
        # cat_emb = th.cat([
        #     self.item_embedding(iid), pos_emb,
        #     self.cate_embedding(g.nodes['i'].data['cate'])
        # ],
        #     dim=1)
        cat_emb = th.cat([
            self.item_embedding(iid),
            self.cate_embedding(g.nodes['i'].data['cate'])
        ],
            dim=1)
        g.nodes['i'].data['f'] = th.matmul(cat_emb, self.W_pos)
        g.nodes['c'].data['f'] = self.cate_embedding(cid)



    def forward(self, g: dgl.DGLHeteroGraph, seq_len, u_input, alias_inputs, adj, mask_item):

        h = self.item_embedding(u_input)
        # local
        h_local = self.local_agg(h, adj, mask_item)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        get = lambda index: h_local[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        h_l = self.handle_local(seq_hidden, mask_item)

        self.feature_encoder(g)

        h = [{
            'i': g.nodes['i'].data['f'],
            'c': g.nodes['c'].data['f']
        }]
        for i, layer in enumerate(self.gnn_layers):
            out = layer(g, (h[-1], h[-1]))
            h.append(out)

        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1, ntype='i')  # index array
        last_cnodes = g.filter_nodes(lambda nodes: nodes.data['clast'] == 1, ntype='c')

        # try gated feat
        feat = self.finalfeature(h)

        h_i = feat['i'][last_nodes].squeeze()  # [bs, embsize]

        h_sum = feat['i'][:seq_len[0]].sum(0).unsqueeze(0)  # 1*embsize
        for j, item in enumerate(seq_len):
            if j > 0:
                tmp = feat['i'][seq_len[j - 1]:seq_len[j]].sum(0)
                h_sum = th.cat([h_sum, tmp.unsqueeze(0)], 0)

        gate = th.sigmoid(th.matmul(th.cat((h_i, h_sum), 1), self.W_hs_e))
        # h_all = gate * h_i + (1 - gate) * h_sum # [bs, embsize]

        h_all = gate * h_i + (1 - gate) * h_sum + h_l  # [bs, embsize]

        feat_last_cate = feat['c'][last_cnodes].squeeze()
        item_embeddings = self.item_embedding.weight[1:]
        item_score = th.matmul(h_all, item_embeddings.t())

        cate_embeddings = self.cate_embedding.weight[1:]
        cate_score = th.matmul(feat_last_cate, cate_embeddings.t())

        return item_score, cate_score, g.batch_num_nodes('i')



class FeatureSelect(nn.Module):
    def __init__(self, embedding_dim, type='last'):
        super().__init__()
        self.embedding_dim = embedding_dim
        assert type in ['last', 'mean', 'gated']
        self.type = type

        self.W_g1 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.W_g2 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.W_g3 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

    def forward(self, h):
        h[0]['i'] = h[0]['i'].squeeze()
        h[-1]['i'] = h[-1]['i'].squeeze()
        h[0]['c'] = h[0]['c'].squeeze()
        h[-1]['c'] = h[-1]['c'].squeeze()
        feature = None
        if self.type == 'last':
            feature = h[-1]
        elif self.type == 'gated':
            gate = th.sigmoid(self.W_g1(th.cat([h[0]['i'], h[-1]['i']], dim=-1)))
            ifeature = gate * h[0]['i'] + (1 - gate) * h[-1]['i']

            gate = th.sigmoid(self.W_g3(th.cat([h[0]['c'], h[-1]['c']], dim=-1)))
            cfeature = gate * h[0]['c'] + (1 - gate) * h[-1]['c']

            feature = {'i': ifeature, 'c': cfeature}

        elif self.type == 'mean':
            isum = th.zeros_like(h[0]['i'])
            csum = th.zeros_like(h[0]['c'])
            for data in h:
                isum += data['i']
                csum += data['c']
            feature = {'i': isum / len(h), 'c': csum / len(h)}

        return feature


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    bgs, label, next_cate, seq_len, u_input, alias_inputs, adj, mask = data

    u_input = trans_to_cuda(u_input).long()
    alias_inputs = trans_to_cuda(alias_inputs).long()
    # alias_category = trans_to_cuda(alias_category).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    seq_len = trans_to_cuda(seq_len).long()
    bgs = bgs.to(th.device('cuda:0' if th.cuda.is_available() else 'cpu'))


    item_score, cate_score, session_length = model.forward(bgs, seq_len, u_input, alias_inputs, adj, mask)
    # item_score = item_score * mask
    # cate_score = cate_score * mask
    return item_score, cate_score,label,next_cate


def train_test(model, train_loader, test_loader):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        item_score, cate_score,label,next_cate = forward(model, data)
        label = trans_to_cuda(label).long()
        next_cate = trans_to_cuda(next_cate).long()
        loss_item = model.loss_function(item_score, label - 1)
        loss_cate = model.loss_function(cate_score, next_cate - 1)
        loss_item += loss_cate

        loss_item.backward()
        model.optimizer.step()

        total_loss +=loss_item

    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()

    result = []
    hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = [], [], [], [], [], [], [], [], [], []

    for data in test_loader:
        scores, cate_score,targets,next_cate = forward(model, data)
        sub_scores_k20 = scores.topk(20)[1]
        sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
        sub_scores_k10 = scores.topk(10)[1]
        sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()

        sub_scores_k30 = scores.topk(30)[1]
        sub_scores_k30 = trans_to_cpu(sub_scores_k30).detach().numpy()
        sub_scores_k40 = scores.topk(40)[1]
        sub_scores_k40 = trans_to_cpu(sub_scores_k40).detach().numpy()
        sub_scores_k50 = scores.topk(50)[1]
        sub_scores_k50 = trans_to_cpu(sub_scores_k50).detach().numpy()
        targets = targets.numpy()

        for score, target, mask in zip(sub_scores_k20, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k20.append(0)
            else:
                mrr_k20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k10, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k10.append(0)
            else:
                mrr_k10.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k30, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k30.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k30.append(0)
            else:
                mrr_k30.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k40, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k40.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k40.append(0)
            else:
                mrr_k40.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k50, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k50.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k50.append(0)
            else:
                mrr_k50.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit_k10) * 100)
    result.append(np.mean(mrr_k10) * 100)
    result.append(np.mean(hit_k20) * 100)
    result.append(np.mean(mrr_k20) * 100)

    result.append(np.mean(hit_k30) * 100)
    result.append(np.mean(mrr_k30) * 100)
    result.append(np.mean(hit_k40) * 100)
    result.append(np.mean(mrr_k40) * 100)
    result.append(np.mean(hit_k50) * 100)
    result.append(np.mean(mrr_k50) * 100)

    return result


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads=1,
                 feat_drop=0.1,
                 attn_drop=0.1,
                 negative_slope=0.2,
                 residual=True,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats,
                                    out_feats * num_heads,
                                    bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats,
                                    out_feats * num_heads,
                                    bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats,
                                out_feats * num_heads,
                                bias=False)
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats,
                                        num_heads * out_feats,
                                        bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    pass
                    # raise DGLError('There are 0-in-degree nodes in the graph, '
                    #                'output for those nodes will be invalid. '
                    #                'This is harmful for some applications, '
                    #                'causing silent performance regression. '
                    #                'Adding self-loop on the input graph by '
                    #                'calling `g = dgl.add_self_loop(g)` will resolve '
                    #                'the issue. Setting ``allow_zero_in_degree`` '
                    #                'to be `True` when constructing this module will '
                    #                'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads,
                                                   self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads,
                                                   self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads,
                                                       self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads,
                                                       self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0],
                                                 self._num_heads,
                                                 self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
