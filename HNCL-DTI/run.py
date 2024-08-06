from itertools import tee
import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import data_loader
from data import load_data
from pytorchtools import EarlyStopping
import argparse
import time
from collections import defaultdict
from info_nce import InfoNCE


ap = argparse.ArgumentParser("HNCL-DTI")
ap.add_argument('--feats-type', type=int, default=2)
ap.add_argument('--hidden_dim', type=int, default=16, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=2, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--batch_size', type=int, default=1024)
ap.add_argument('--patience', type=int, default=40, help='Patience.')
ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=5e-3)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--slope', type=float, default=0.1)
ap.add_argument('--dataset', type=str, default='HBN-A')
ap.add_argument('--decode', type=str, default='dot')
ap.add_argument('--gamma', type=float, default=0.01)
ap.add_argument('--edge-feats', type=int, default=16)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--use_norm', type=bool, default=True)
ap.add_argument('--residual-att', type=float, default=0.)
ap.add_argument('--residual', type=bool, default=True)
ap.add_argument('--run', type=int, default=1)

args = ap.parse_args()
device = torch.device("cuda:"+str(args.device))

def DistMult(num_rel, dim, left_emb, right_emb, r_id):
    W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
    nn.init.xavier_normal_(W, gain=1.414)
    thW = W[r_id].cuda()
    left_emb = torch.unsqueeze(left_emb, 1)
    right_emb = torch.unsqueeze(right_emb, 2)
    return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

def Dot(left_emb, right_emb, r_id):
    left_emb = torch.unsqueeze(left_emb, 1)
    right_emb = torch.unsqueeze(right_emb, 2)
    return torch.bmm(left_emb, right_emb).squeeze()


def meta2str(meta_tuple):
    return str(meta_tuple[0]) + '_' + str(meta_tuple[1])

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

# dataset = data_loader('../'+args.dataset)
features_list, adjM, dl = load_data(args.dataset)
edge_dict = {}
# edge_type2meta = {}
for i, meta_path in dl.links['meta'].items():
    edge_dict[(str(meta_path[0]), str(i), str(meta_path[1]))] = (torch.tensor(dl.links['data'][i].tocoo().row - dl.nodes['shift'][meta_path[0]]), torch.tensor(dl.links['data'][i].tocoo().col - dl.nodes['shift'][meta_path[1]]))
    # edge_type2meta[i] = str(i) #str(meta_path[0]) + '_' + str(meta_path[1])


node_count = {}
for i, count in dl.nodes['count'].items():
    print(i, node_count)
    node_count[str(i)] = count

G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
"""
for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
    # print(G.nodes['attr'][ntype].shape)
"""

G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    print(ntype, dl.nodes['shift'][int(ntype)])
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 


feats_type = args.feats_type
# features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
features_list = [mat2tensor(features).to(device) for features in features_list]

if feats_type == 0:
    in_dims = [features.shape[1] for features in features_list]
elif feats_type == 1 or feats_type == 5:
    save = 0 if feats_type == 1 else 2
    in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
    for i in range(0, len(features_list)):
        if i == save:
            in_dims.append(features_list[i].shape[1])
        else:
            in_dims.append(10)
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
elif feats_type == 2 or feats_type == 4:
    save = feats_type - 2
    in_dims = [features.shape[0] for features in features_list]
    for i in range(0, len(features_list)):
        if i == save:
            in_dims[i] = features_list[i].shape[1]
            continue
        dim = features_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
elif feats_type == 3:
    in_dims = [features.shape[0] for features in features_list]
    for i in range(len(features_list)):
        dim = features_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

edge2type = {}
for k in dl.links['data']:
    for u,v in zip(*dl.links['data'][k].nonzero()):
        edge2type[(u,v)] = k
for i in range(dl.nodes['total']):
    if (i,i) not in edge2type:
        edge2type[(i,i)] = len(dl.links['count'])
for k in dl.links['data']:
    for u,v in zip(*dl.links['data'][k].nonzero()):
        if (v,u) not in edge2type:
            edge2type[(v,u)] = k+1+len(dl.links['count'])

g = dgl.DGLGraph(adjM+(adjM.T))
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
g = g.to(device)
e_feat = []
for u, v in zip(*g.edges()):
    u = u.cpu().item()
    v = v.cpu().item()
    e_feat.append(edge2type[(u,v)])
e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
links_test = [2]



for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = features_list[int(ntype)]#.to(device)

train_pos, valid_pos, test_pos = dl.get_train_valid_test_pos()
early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_HNCL.pt'.format(args.dataset))
train_step = 0
# loss_func = torch.nn.BCELoss() ### loss_func
train_pos_head_full = np.array([])
train_pos_tail_full = np.array([])
train_neg_head_full = np.array([])
train_neg_tail_full = np.array([])
r_id_full = []
test_edge_type = links_test[0]
train_neg = dl.get_train_neg()[test_edge_type]
train_pos_head_full = np.concatenate([train_pos_head_full, np.array(train_pos[test_edge_type][0])])
train_pos_tail_full = np.concatenate([train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
train_neg_head_full = np.concatenate([train_neg_head_full, np.array(train_neg[0])])
train_neg_tail_full = np.concatenate([train_neg_tail_full, np.array(train_neg[1])])
# r_id_full = np.concatenate([r_id_full, np.array([test_edge_type]*len(train_pos[test_edge_type][0]))])
r_id_full.extend([int(test_edge_type)]* len(train_pos[test_edge_type][0]))

num_classes = args.hidden_dim
heads = [args.num_heads] * (args.num_layers) + [args.num_heads]#[1]
net = HNCLE(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, args.residual, args.residual_att, decode=args.decode)
optimizer_HNCLE = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
net.to(device)

model = HNCLN(G, n_inps=in_dims, n_hid=args.hidden_dim*(args.num_layers+2), n_layers=args.num_layers, n_heads=args.num_heads, use_norm = args.use_norm, decode=args.decode).to(device)
optimizer_HNCLN = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#, lr=args.lr, weight_decay=args.weight_decay)

# total = len(list(dl.links_test['data'].keys()))
r_id_full = np.asarray(r_id_full)
train_step = 0 
first_flag = True

for epoch in range(args.epoch):
    train_idx = np.arange(len(train_pos_head_full))
    np.random.shuffle(train_idx)
    batch_size = len(train_pos_head_full)#len(train_pos_head_full)#args.batch_size
    # batch_size = args.batch_size
    for step, start in enumerate(range(0, len(train_pos_head_full), batch_size)):
        contrast = InfoNCE()
        t_start = time.time()
        model.train()
        net.train()
        train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
        train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
        train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
        train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
        r_id = r_id_full[train_idx[start:start+batch_size]]       
        left = np.concatenate([train_pos_head, train_neg_head])
        right = np.concatenate([train_pos_tail, train_neg_tail])
        mid = np.concatenate([r_id, r_id])

        train_true_edges = np.array([train_pos_head, train_pos_tail])
        train_false_edges = np.array([train_neg_head, train_neg_tail])
        # left_shift = np.asarray([dl.nodes['shift'][x] for x in mid[:,0]])
        # right_shift = np.asarray([dl.nodes['shift'][x] for x in mid[:,1]])

        logits_HNCLN, emb_HNCLN = model(G, left, right, mid)
        logits_HNCLE, emb_HNCLE = net(features_list, e_feat, left, right, mid)
        emb = (emb_HNCLN + emb_HNCLE)/2
        labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)

        emb_true_first = []
        emb_true_second = []
        emb_false_first = []
        emb_false_second = []

        for i in range(len(train_true_edges)):
            if i == 0:
                for head in train_true_edges[i]:
                    emb_true_first.append(emb[int(head)])
            if i == 1: 
                for tail in train_true_edges[i]:
                    emb_true_second.append(emb[int(tail)])

        for i in range(len(train_false_edges)):
            if i == 0:
                for head in train_false_edges[i]:
                    emb_false_first.append(emb[int(head)])
            if i == 1: 
                for tail in train_false_edges[i]:
                   emb_false_second.append(emb[int(tail)])
        
        emb_true_first = torch.cat(emb_true_first).reshape(-1, args.hidden_dim*(args.num_layers+2))
        emb_true_second = torch.cat(emb_true_second).reshape(-1, args.hidden_dim*(args.num_layers+2))
        emb_false_first = torch.cat(emb_false_first).reshape(-1, args.hidden_dim*(args.num_layers+2))
        emb_false_second = torch.cat(emb_false_second ).reshape(-1, args.hidden_dim*(args.num_layers+2))

        T1 = emb_true_first @ emb_true_second.T
        T2 = -(emb_false_first @ emb_false_second.T)


        pos_out = torch.diag(T1)
        neg_out = torch.diag(T2)
        train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))+ args.gamma * contrast(emb_HNCLE, emb_HNCLN)
        train_loss = train_loss.requires_grad_()

        optimizer_HNCLN.zero_grad()
        optimizer_HNCLE.zero_grad()
        train_loss.backward()
        optimizer_HNCLN.step()
        optimizer_HNCLE.step()
        train_step += 1
        t_end = time.time()
        print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))
        t_start = time.time()
            # validation
        model.eval()
        net.eval()
        with torch.no_grad():
            valid_pos_head = np.array([])
            valid_pos_tail = np.array([])
            valid_neg_head = np.array([])
            valid_neg_tail = np.array([])
            valid_r_id = []
            for test_edge_type in links_test:
                valid_neg = dl.get_valid_neg()[test_edge_type]
                valid_pos_head = np.concatenate([valid_pos_head, np.array(valid_pos[test_edge_type][0])])
                valid_pos_tail = np.concatenate([valid_pos_tail, np.array(valid_pos[test_edge_type][1])])
                valid_neg_head = np.concatenate([valid_neg_head, np.array(valid_neg[0])])
                valid_neg_tail = np.concatenate([valid_neg_tail, np.array(valid_neg[1])])
                valid_r_id.extend([int(test_edge_type)]*len(valid_pos[test_edge_type][0]))
            left = np.concatenate([valid_pos_head, valid_neg_head])
            right = np.concatenate([valid_pos_tail, valid_neg_tail])
            mid = np.concatenate([valid_r_id, valid_r_id])

            valid_true_edges = np.array([valid_pos_head, valid_pos_tail])
            valid_false_edges = np.array([valid_neg_head, valid_neg_tail])
            logits_HNCLN, emb_HNCLN = model(G, left, right, mid)
            logits_HNCLE, emb_HNCLE = net(features_list, e_feat, left, right, mid)
            emb = (emb_HNCLN + emb_HNCLE)/2
            labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)

            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(len(valid_true_edges)):
                if i == 0:
                    for head in valid_true_edges[i]:
                        emb_true_first.append(emb[int(head)])
                if i == 1: 
                    for tail in valid_true_edges[i]:
                        emb_true_second.append(emb[int(tail)])

            for i in range(len(valid_false_edges)):
                if i == 0:
                    for head in valid_false_edges[i]:
                        emb_false_first.append(emb[int(head)])
                if i == 1: 
                    for tail in valid_false_edges[i]:
                        emb_false_second.append(emb[int(tail)])

            emb_true_first = torch.cat(emb_true_first).reshape(-1, args.hidden_dim*(args.num_layers+2))
            emb_true_second = torch.cat(emb_true_second).reshape(-1, args.hidden_dim*(args.num_layers+2))
            emb_false_first = torch.cat(emb_false_first).reshape(-1, args.hidden_dim*(args.num_layers+2))
            emb_false_second = torch.cat(emb_false_second ).reshape(-1, args.hidden_dim*(args.num_layers+2))

            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)
            val_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))+ args.gamma * contrast(emb_HNCLE, emb_HNCLN)
            val_loss = val_loss.requires_grad_()

        t_end = time.time()
        # print validation info
        print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), t_end - t_start))
        # early stopping
        early_stopping(val_loss, model, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    if early_stopping.early_stop:
        print('Early stopping!')
        break

for test_edge_type in links_test:
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}_HNCL.pt'.format(args.dataset))['model'])
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}_HNCL.pt'.format(args.dataset))['net'])
    model.eval()
    net.eval()
    test_logits = []
    with torch.no_grad():
        test_pair, test_label = [], []
        test_pos = test_pos[test_edge_type]
        test_neg = dl.get_test_neg()[test_edge_type]
        test_pair.append(test_pos[0] + test_neg[0])
        test_pair.append(test_pos[1] + test_neg[1])
        test_label = [1]*len(test_pos[0]) + [0]*len(test_neg[0])
        left = np.array(test_pair[0])
        right = np.array(test_pair[1])
        mid = [int(test_edge_type)] * left.shape[0]
        mid = np.asarray(mid)
        labels = torch.FloatTensor(test_label).to(device)
        logits_HNCLN, emb_HNCLN = model(G, left, right, mid)
        logits_HNCLE, emb_HNCLE = net(features_list, e_feat, left, right, mid)
        emb = (emb_HNCLN + emb_HNCLE)/2
        left_emb = emb[left]
        right_emb = emb[right]
        pred = F.sigmoid(Dot(left_emb, right_emb, mid)).cpu().numpy()
        edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)
        labels = labels.cpu().numpy()
        res = dl.evaluate(edge_list, pred, labels)
        print(res)