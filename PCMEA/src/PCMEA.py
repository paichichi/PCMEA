#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function
from module_dic import *
import numpy as np




try:
    from layers import *
except:
    from src.layers import *


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj, g_device):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj, g_device)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


""" vanilla GCN """


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(x)
        return x


""" loss """


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb


class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):

        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)

        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):

        return torch.matmul(query, key.transpose(-1, -2))

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        seq_len, _ = query.shape  # 27793

        query = self.query(query) # torch.Size([27793, 10, 10])
        key = self.key(key)  # torch.Size([27793, 10, 10])
        value = self.value(value) # torch.Size([27793, 10, 10])

        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        x = torch.matmul(attn, value.transpose(-1, -2))

        self.attn = attn.detach()

        x = x.reshape(seq_len, -1)

        # Output layer
        return self.output(x)

class FeedForward(Module):
    """
    ## FFN module
    """
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):

        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):

        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer

    def forward(self, *,
                x: torch.Tensor, y:torch.Tensor):

        z = self.norm_self_attn(x)  # torch.Size([27793, 100])
        m,n = z.shape
        m2,n2 = y.shape
        y2 = torch.unsqueeze(y,dim=0)
        z2 = F.interpolate(y2, scale_factor=n / n2)
        z2 = z2.squeeze()

        self_attn = self.self_attn(query=z2, key=z2, value=z)
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)
        return x



class MultiModalEncoder_addAtt_addRel(nn.Module):
    """
    entity embedding: (ent_num, input_dim)
    gcn layer: n_units

    """

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False):
        super(MultiModalEncoder_addAtt_addRel, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim  # 100
        # add
        att_text_dim = self.args.att_text_dim  # 100 ## TODO
        rel_text_dim = self.args.rel_text_dim  # 100 ## TODO

        img_dim = self.args.img_dim  # 100
        char_dim = self.args.char_dim  # 100
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        ######## Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        #########################
        ######## Modal Encoder
        #########################

        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(1000, attr_dim)
        # add
        self.att_text_fc = nn.Linear(768, att_text_dim) ##TODO:new add
        self.rel_text_fc = nn.Linear(768, rel_text_dim)  ##TODO:new add

        self.img_fc = nn.Linear(img_feature_dim, img_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)
        #########################
        ######## Fusion Encoder
        #########################

        self.c_mhsa_mlp = Cross_Att_With_mlp(d_model=img_dim, dropout=0.1)
        self.c_mhsa_mlp1 = Cross_Att_With_mlp(d_model=img_dim, dropout=0.1)
        self.s_adaptormlp = Self_att_With_Adaptor(d_model=img_dim * 3, dropout=0.1)


        self.fusion = MultiModalFusion(modal_num=self.args.inner_view_num,
                                       with_weight=self.args.with_weight)
        self.fusion1 = MultiModalFusion(modal_num=self.args.inner_view_num,
                                       with_weight=self.args.with_weight)

    def forward(self, e_device, input_idx,adj,
                img_features=None,
                rel_features=None,
                att_features=None,
                att_text_features=None,
                rel_text_features=None):

        if self.args.w_gcn:

            gph_emb = self.cross_graph_model(self.entity_emb(input_idx).to(e_device), adj, e_device)
        else:
            gph_emb = None
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None

        if self.args.w_attr_text:
            att_text_emb = self.att_text_fc(att_text_features)

        if self.args.w_rel_text:
            rel_text_emb = self.rel_text_fc(rel_text_features)

        joint_emb0 = self.fusion1([img_emb, att_emb, rel_emb, gph_emb, att_text_emb, rel_text_emb])

        gph_emb = self.s_adaptormlp(gph_emb)

        rel_emb = self.c_mhsa_mlp(uni_emb= rel_emb, cross_emb= rel_text_emb)
        att_emb = self.c_mhsa_mlp1(uni_emb= att_emb, cross_emb=att_text_emb)

        joint_emb1 = self.fusion([img_emb, att_emb, rel_emb, gph_emb, att_text_emb, rel_text_emb])

        return gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, joint_emb1, joint_emb0


class MINE(nn.Module):
    def __init__(self, data_dim, hidden_size=10):
        super(MINE, self).__init__()

        self.layers = nn.Sequential(nn.Linear(data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x, y):

        inputs = torch.cat([x, y], dim=1)
        logits = self.layers(inputs)

        loss = - np.log2(np.exp(1)) * (torch.mean(logits) - torch.log(torch.mean(torch.exp(logits))))
        return loss


