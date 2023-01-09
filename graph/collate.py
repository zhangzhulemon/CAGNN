
import dgl
import torch


def gnn_collate_fn(seq_to_graph_fn):

    def collate_fn(data):
        # adj,u_input,alias_inputs,category_ids,mask,seq_len,label,next_cate
        adj, u_input, alias_inputs,category_ids, mask, seq_len, label, next_cate = zip(*data)

        data_g = zip(u_input, category_ids, seq_len)
        graphs = list(map(seq_to_graph_fn, data_g))
        bgs = dgl.batch(graphs)

        seq_len_new = [sum(seq_len[:j + 1]) for j, len in enumerate(seq_len)]

        # return bgs, label, next_cate, seq_len_new,u_input, alias_inputs, adj, mask
        return [bgs, torch.tensor(label), torch.tensor(next_cate),
                torch.tensor(seq_len_new), torch.stack(u_input), torch.stack(alias_inputs),torch.stack(adj),
                torch.stack(mask)]


    return collate_fn
