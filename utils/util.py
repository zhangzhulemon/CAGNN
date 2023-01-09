import numpy as np
import torch
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len


def getCaIds(input, category):
    return [0 if item == 0 else category[item] for item in input]


class Data(Dataset):
    def __init__(self, data, category, train_len=None):
        inputs, mask, max_len = handle_data(data[0], train_len)
        self.category = category
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.seq_len = np.asarray([len(x) for x in data[0]])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, label, seq_len = self.inputs[index], self.mask[index], self.targets[index], self.seq_len[index]
        category_ids= getCaIds(u_input, self.category)
        # ________________________________________________________________
        max_n_node = self.max_len
        node = np.unique(u_input)
        adj = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        # node_cat = np.unique(category_ids)
        # alias_category = [np.where(node_cat == i)[0][0] for i in category_ids]

        next_cate = self.category[label]
        # alias_inputs,alias_category,adj,u_input,category_ids,mask,seq_len,label,next_cate
        return [torch.tensor(adj), torch.tensor(u_input), torch.tensor(alias_inputs),torch.tensor(category_ids),torch.tensor(mask),
                torch.tensor(seq_len), torch.tensor(label), torch.tensor(next_cate)]


    def __len__(self):
        return self.length



def handle_category(category):
    cate_uniq = np.unique(list(category.values()))

    category_new = {}
    for item, category in category.items():
        category_new[item] = np.where(cate_uniq == category)[0][0] + 1
    return category_new







def data_statistics(train, test, categorys):
    seqs = train[0] + test[0]
    labels = train[1] + test[1]

    items = set()
    cats = set()
    total_session_length = 0
    total_cat_per_session = 0

    for x in seqs:
        total_session_length += len(x)
        total_cat_per_session += len(np.unique([categorys[i] for i in x]))

        items.update(x)
        cats.update([categorys[i] for i in x ])

    items.update(labels)
    for i in labels:
        cats.add(categorys[i])



    print('')
    print('*******dataset statistics:*******')
    print('=============================================')
    print('No. of items: {}'.format(len(items)))
    print('No. of sessions: {}'.format(len(seqs)))
    print('Avg. of session length: {}'.format(total_session_length / len(seqs)))
    print('No. of categories: {}'.format(len(cats)))
    print('No. of cats/session: {}'.format(total_cat_per_session / len(seqs)))
    print('min item: {}'.format(min(items)))
    print('max item: {}'.format(max(items)))
    print('min cate: {}'.format(min(cats)))
    print('max cate: {}'.format(max(cats)))
    print('=============================================')
