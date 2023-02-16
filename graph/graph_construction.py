
from typing import List, Tuple

import dgl
import numpy as np
import torch


def label_to_hetero_graph(g: dgl.DGLHeteroGraph, items, cates, map_c_items, positions, last_id, last_cid):

    g.nodes["i"].data['id'] = torch.from_numpy(items)
    g.nodes['c'].data['id'] = torch.from_numpy(cates)
    # g.edges['c2i'].data['pos'] = torch.from_numpy(positions)

    i_count = len(items)
    c_count = len(cates)

    items_cat_label = torch.zeros(i_count, dtype=torch.int32)
    for c, c_items in map_c_items.items():
        items_cat_label[c_items] = c
    g.nodes['i'].data['cate'] = items_cat_label ##zz: 设置'i'类型的节点的'cate'特征

    last_id_label = torch.zeros(i_count, dtype=torch.int32)
    last_id_label[last_id] = 1
    g.nodes['i'].data['last'] = last_id_label
    
    last_cid_label = torch.zeros(c_count, dtype=torch.int32)
    last_cid_label[last_cid] = 1
    g.nodes['c'].data['clast'] = last_cid_label

    return g


def seq_to_hetero_graph(data: Tuple[List, List, int, int]) -> dgl.DGLHeteroGraph:

    item_ids, cate_ids, seq_len = data
    item_ids = item_ids[:seq_len]
    cate_ids = cate_ids[:seq_len]
    e_i2i = [np.array([], dtype='int'), np.array([], dtype='int')]
    e_c2c = [0, 0]

    positions = np.array([seq_len - 1 - _ for _ in range(seq_len)], dtype='float')
    cats, cat_nid = np.unique(cate_ids, return_inverse=True)
    items, item_nid = np.unique(item_ids, return_inverse=True)


    cat_nid_u = np.unique(cat_nid)

    map_c_items = {}

    for c in cat_nid_u:
        mask = cat_nid == c
        item_nid_in_c = item_nid[mask]

        if len(item_nid_in_c) > 0:
            # item-item sequence and item self-loop
            e_i2i[0] = np.concatenate((e_i2i[0], item_nid_in_c[:-1], item_nid_in_c), axis=0)
            e_i2i[1] = np.concatenate((e_i2i[1], item_nid_in_c[1:], item_nid_in_c), axis=0)

            map_c_items[c] = item_nid_in_c

    last_id = item_nid[-1]
    last_cid = cat_nid[-1]



    # category-category sequence and self-loop
    e_c2c[0] = np.concatenate((cat_nid[:-1], cat_nid))
    e_c2c[1] = np.concatenate((cat_nid[1:], cat_nid))



    # construct the graph
    # zz （初始节点类型，边的类型，终止节点类型）
    g = dgl.heterograph({("i", "i2i", "i"): tuple(e_i2i),
                         ('i', 'i2c', 'c'): (item_nid, cat_nid),
                         ('c', 'c2i', 'i'): (cat_nid, item_nid),
                         ('c', 'c2c', 'c'): tuple(e_c2c)
                         },
                        num_nodes_dict={'i': len(items), 'c': len(cats)})

    g = label_to_hetero_graph(g, items, cats, map_c_items, positions, last_id, last_cid)
    
    return g


