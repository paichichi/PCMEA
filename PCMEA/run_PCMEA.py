#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import os

from src.utils import read_raw_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import argparse
from pprint import pprint

import torch.optim as optim

from src.PCMEA import *
from src.Load import *
from PCMEA import *
from src.module_dic import *

model_name = "PCMEA"
print("now in training is :" + model_name)


acc_top1_list = []
acc_epoch_list = []

pred_link_ratio = []
pred_link_num = []
pred_wrong_num = []
pred_epoch_list = []


def load_img_features(ent_num, file_dir):
    # load images features
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = "your path" + filename + "/" + filename + "_id_img_feature_dict.pkl"
    else:
        img_vec_path = None

    img_features = load_img(ent_num, img_vec_path)
    return img_features


def load_att_txt_features(ent_num, file_dir):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        att_text_vec_path = "Your path" + filename + "/" + "attribute_roberta_base_feature_all.pkl" # roberta-base
        att_txt_features = load_att_text(ent_num, att_text_vec_path)
    else:
        att_txt_features = None
    return att_txt_features



def load_rel_txt_features(ent_num, file_dir):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        rel_text_vec_path = "your path" + filename + "/" + "relation_t5_feature_all.pkl"
    else:
        pass

    rel_txt_features = load_rel_text(ent_num, rel_text_vec_path)
    return rel_txt_features


class PCMEA:

    def __init__(self):

        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None

        self.startEpoch = 0

        self.img_features = None
        self.rel_features = None
        self.att_features = None
        self.ent_vec = None

        self.att_txt_features = None ##TODO
        self.rel_txt_features = None ##TODO

        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.adj = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None

        self.train_list = None

        # model
        self.multimodal_encoder = None

        self.momentum_encoder = None
        self.m = 0.999

        self.dyn_link = {} ##TODO:New

        self.mine_vj = None
        self.mine_aj = None
        self.mine_rj = None
        self.mine_gj = None

        self.weight_raw = None
        self.rel_fc = None
        self.att_fc = None
        self.img_fc = None
        self.shared_fc = None
        self.att_text_fc = None ##TODO
        self.rel_text_fc = None ##TODO

        self.gcn_pro = None
        self.rel_pro = None
        self.attr_pro = None
        self.img_pro = None
        self.att_text_pro = None ##TODO
        self.rel_text_pro = None ##TODO


        self.input_dim = None
        self.entity_emb = None
        self.join_mon_emb = None


        self.input_idx = None
        self.n_units = None
        self.n_heads = None
        self.cross_graph_model = None
        self.params = None
        self.optimizer = None

        self.criterion_cl = None
        self.criterion_cl_2 = None  ##TODO
        self.criterion_align = None

        self.multi_loss_layer = None
        self.align_multi_loss_layer = None
        self.fusion = None

        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)

        self.set_seed(self.args.seed, self.args.cuda)

        self.device = torch.device("cuda:0" if self.args.cuda and torch.cuda.is_available() else "cpu")
        self.device2 = torch.device("cuda:1" if self.args.cuda and torch.cuda.is_available() else "cpu")

        self.init_data()
        self.init_model()

        self.print_summary()

    @staticmethod
    def parse_options(parser):
        parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False,
                            help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")

        parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument("--hidden_units", type=str, default="128,128,128",
                            help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False,
                            help="enable instance normalization")
        parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode")
        parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")
        # parser.add_argument("--long_tail_analysis", action="store_true", default=False)
        parser.add_argument("--lta_split", type=int, default=0, help="split in {0,1,2,3,|splits|-1}")
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--tau2", type=float, default=1, help="the temperature factor of alignment loss")
        parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different "
                                                                       "modal features")
        parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view, "
                                                                                 "[gcn|gat]")

        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")

        parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")

        parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
        parser.add_argument("--img_dim", type=int, default=100, help="the hidden size of img feature")
        parser.add_argument("--att_text_dim", type=int, default=100, help="the hidden size of attr-text features") ##TODO
        parser.add_argument("--rel_text_dim", type=int, default=100, help="the hidden size of rel-text features") ##TODO

        parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_img", action="store_false", default=True, help="with img features")
        parser.add_argument("--w_attr_text", action="store_false", default=True, help="with att text features") ##TODO:new add
        parser.add_argument("--w_rel_text", action="store_false", default=True,
                            help="with rel text features")  ##TODO:new add

        parser.add_argument("--inner_view_num", type=int, default=7, help="the number of inner view") ##TODO: new modify

        parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, "
                                                                                "[glove|fasttext]")
        parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")

        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]")
        parser.add_argument("--save_path", type=str, default="save_pkl", help="save path")

        return parser.parse_args()

    @staticmethod
    def set_seed(seed, cuda=True):
        # random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def print_summary(self):
        print("-----dataset summary-----")
        print("dataset:\t", self.args.file_dir)
        print("triple num:\t", len(self.triples))
        print("entity num:\t", self.ENT_NUM)
        print("relation num:\t", self.REL_NUM)
        print("train ill num:\t", self.train_ill.shape[0], "\ttest ill num:\t", self.test_ill.shape[0])
        print("-------------------------")

    def init_data(self):
        # Load data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device

        self.ent2id_dict, self.ills, self.triples, self.r_hs, self.r_ts, self.ids = read_raw_data(file_dir, lang_list)
        e1 = os.path.join(file_dir, 'ent_ids_1')
        e2 = os.path.join(file_dir, 'ent_ids_2')
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)

        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.r_hs)
        print("total ent num: {}, rel num: {}".format(self.ENT_NUM, self.REL_NUM))

        np.random.shuffle(self.ills)

        # load images features
        self.img_features = load_img_features(self.ENT_NUM, file_dir)
        self.img_features = F.normalize(torch.Tensor(self.img_features).to(device))
        print("image feature shape:", self.img_features.shape)

        # load att text features
        # Todo: new add
        self.att_txt_features = load_att_txt_features(self.ENT_NUM, file_dir)
        self.att_txt_features = F.normalize(torch.Tensor(self.att_txt_features).to(device))
        print("att_txt_features shape:", self.att_txt_features.shape)

        # load rel text features
        # Todo: new add
        self.rel_txt_features = load_rel_txt_features(self.ENT_NUM, file_dir)
        self.rel_txt_features = F.normalize(torch.Tensor(self.rel_txt_features).to(device))
        print("rel_txt_features shape:", self.rel_txt_features.shape)

        # train/val/test split
        self.train_ill = np.array(self.ills[:int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32)

        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate):]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)

        self.left_non_train = list(set(self.left_ents) - set(self.train_ill[:, 0].tolist()))
        self.right_non_train = list(set(self.right_ents) - set(self.train_ill[:, 1].tolist()))

        print("#left entity : %d, #right entity: %d" % (len(self.left_ents), len(self.right_ents)))
        print("#left entity not in train set: %d, #right entity not in train set: %d"
              % (len(self.left_non_train), len(self.right_non_train)))

        # convert relations to numbers
        self.rel_features = load_relation(self.ENT_NUM, self.triples, 1000)
        self.rel_features = torch.Tensor(self.rel_features).to(device)
        print("relation feature shape:", self.rel_features.shape)

        a1 = os.path.join(file_dir, 'training_attrs_1')
        a2 = os.path.join(file_dir, 'training_attrs_2')
        self.att_features = load_attr([a1, a2], self.ENT_NUM, self.ent2id_dict, 1000)  # attr
        self.att_features = torch.Tensor(self.att_features).to(device)
        print("attribute feature shape:", self.att_features.shape)

        self.adj = get_adjr(self.ENT_NUM, self.triples, norm=True)  # getting a sparse tensor r_adj
        self.adj = self.adj.to(self.device)

    def init_model(self):
        img_dim = self.img_features.shape[1]
        char_dim = self.char_features.shape[1] if self.char_features is not None else 100
        self.multimodal_encoder = MultiModalEncoder_addAtt_addRel(args=self.args,
                                                    ent_num=self.ENT_NUM,
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=char_dim,
                                                    use_project_head=self.args.use_project_head).to(self.device)

        self.mine_vj = MINE(data_dim=char_dim + char_dim * 8).to(self.device)
        self.mine_aj = MINE(data_dim=char_dim + char_dim * 8).to(self.device)
        self.mine_rj = MINE(data_dim=char_dim + char_dim * 8).to(self.device)
        self.mine_gj = MINE(data_dim=char_dim * 3 + char_dim * 8).to(self.device)

        print("init the model class")

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=self.args.inner_view_num).to(self.device)
        self.align_multi_loss_layer = CustomMultiLossLayer(loss_num=self.args.inner_view_num).to(self.device)

        self.params = [
            {"params":
                 list(self.multimodal_encoder.parameters()) +
                 list(self.multi_loss_layer.parameters()) +
                 list(self.align_multi_loss_layer.parameters())
             }]
        self.optimizer = optim.AdamW(
            self.params,
            lr=self.args.lr
        )
        total_params = sum(p.numel() for p in self.multimodal_encoder.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.multi_loss_layer.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.align_multi_loss_layer.parameters() if p.requires_grad)

        # load model
        name = model_name
        path = r'weight'
        model_save_path = os.path.join(path, name)
        if os.path.exists(model_save_path):
            if os.listdir(model_save_path):
                weight_lt = os.listdir(model_save_path)
                weight_lt.sort()
                weight_lt.sort(key=lambda x: int(x[:-4]))
                self.startEpoch = int(weight_lt[-1][:-4])
                checkpoint = torch.load(os.path.join(model_save_path, weight_lt[-1]))
                self.multimodal_encoder.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("now path is:")
        print(os.path.abspath('src'))

        print("total params num", total_params)
        print("model details:")
        print(self.multimodal_encoder.cross_graph_model)
        print("optimiser details:")
        print(self.optimizer)

        self.criterion_cl = icl_loss(device=self.device, tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_cl_2 = icl_loss(device=self.device, tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        self.criterion_align = ial_loss(device=self.device, tau=self.args.tau2,
                                        ab_weight=self.args.ab_weight,
                                        zoom=self.args.zoom,
                                        reduction=self.args.reduction)


    def momentum_emb(self):
        img_dim = self.img_features.shape[1]
        char_dim = self.char_features.shape[1] if self.char_features is not None else 100
        if self.momentum_encoder == None:
            self.momentum_encoder = MultiModalEncoder_addAtt_addRel(args=self.args,
                                                        ent_num=self.ENT_NUM,
                                                        img_feature_dim=img_dim,
                                                        char_feature_dim=char_dim,
                                                        use_project_head=self.args.use_project_head).to(self.device2)
            for param_q, param_k in zip(self.multimodal_encoder.parameters(), self.momentum_encoder.parameters()):
                param_k.data = param_q.data.to(self.device2)
        else:
            for param_q, param_k in zip(self.multimodal_encoder.parameters(), self.momentum_encoder.parameters()):
                param_k.data = (param_q.data.to(self.device2) * (1. - self.m) + param_k.data * self.m)

        del param_q, param_k
        torch.cuda.empty_cache()
        gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb,joint_emb, _ = self.momentum_encoder(self.device2,
                                                                 self.input_idx.to(self.device2),
                                                                 self.adj.to(self.device2),
                                                                 self.img_features.to(self.device2),
                                                                 self.rel_features.to(self.device2),
                                                                 self.att_features.to(self.device2),
                                                                 att_text_features=self.att_txt_features.to(self.device2),
                                                                 rel_text_features=self.rel_txt_features.to(self.device2),
                                                       )
        print("Using momentum network generate emb....")

        final_emb = F.normalize(joint_emb)

        del gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb,
        torch.cuda.empty_cache()

        return final_emb


    def semi_supervised_learning(self):

        ##TODO: addition of momentum encoder to generate pred link
        with torch.no_grad():

            gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, \
             joint_emb,_ = self.multimodal_encoder(self.device,self.input_idx,self.adj,
                                                                    self.img_features,self.rel_features,
                                                                    self.att_features,att_text_features=self.att_txt_features,
                                                                    rel_text_features = self.rel_txt_features)

            final_emb = F.normalize(joint_emb)

        ##TODO:only compare one-moment model
        distance_list = []
        for i in np.arange(0, len(self.left_non_train), 1000):
            d = pairwise_distances(final_emb[self.left_non_train[i:i + 1000]], final_emb[self.right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        return preds_l, preds_r



    def inner_view_loss(self, gph_emb, rel_emb, att_emb, att_text_emb, rel_text_emb,
                        img_emb, train_ill):

        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_att_text = self.criterion_cl(att_text_emb, train_ill) if att_text_emb is not None else 0 ##TODO
        loss_rel_text = self.criterion_cl(rel_text_emb, train_ill) if rel_text_emb is not None else 0 ##TODO

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_att_text, loss_rel_text, loss_img])
        return total_loss

    def kl_alignment_loss(self, joint_emb, gph_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, img_emb, train_ill):

        zoom = self.args.zoom
        loss_GCN = self.criterion_align(gph_emb, joint_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_align(rel_emb, joint_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_align(att_emb, joint_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_align(img_emb, joint_emb, train_ill) if img_emb is not None else 0
        loss_att_text = self.criterion_align(att_text_emb, joint_emb, train_ill) if att_text_emb is not None else 0 ##TODO
        loss_rel_text = self.criterion_align(rel_text_emb, joint_emb, train_ill) if rel_text_emb is not None else 0 ##TODO

        total_loss = self.align_multi_loss_layer(
                [loss_GCN, loss_rel, loss_att, loss_att_text, loss_rel_text, loss_img]) * zoom
        return total_loss

    def train(self):

        # print args
        pprint(self.args)

        # Train
        print("[start training...] ")
        t_total = time.time()
        new_links = []
        epoch_KE, epoch_CG = 0, 0

        bsize = self.args.bsize
        device = self.device

        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)
        print(self.startEpoch)
        print(self.args.epochs)

        with open(r"{}_true_link_note.txt".format(model_name), 'a') as f:
            f.write(str(model_name))
            f.write(':\n')

        with open(r"{}_accuracy_note.txt".format(model_name), 'a') as f:
            f.write(str(model_name))
            f.write(':\n')

        for epoch in range(self.startEpoch, self.args.epochs):

            if epoch == epoch >= self.args.il_start:
                self.optimizer = optim.AdamW(self.params, lr=self.args.lr / 5)

            t_epoch = time.time()

            self.multimodal_encoder.train()

            # TODO: uni-modal vs joint-modal
            self.mine_vj.train()
            self.mine_rj.train()
            self.mine_gj.train()
            self.mine_aj.train()


            self.multi_loss_layer.train()
            self.align_multi_loss_layer.train()
            self.optimizer.zero_grad()

            gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, \
            joint_emb, joint_emb0 = self.multimodal_encoder(self.device, self.input_idx, self.adj,
                                                                    self.img_features, self.rel_features,
                                                                    self.att_features,
                                                                    att_text_features=self.att_txt_features,
                                                                    rel_text_features=self.rel_txt_features)
            nce_v = self.mine_vj(img_emb.cuda(), joint_emb.cuda())
            nce_a = self.mine_aj(att_text_emb.cuda(), joint_emb.cuda())
            nce_r = self.mine_rj(rel_text_emb.cuda(), joint_emb.cuda())
            nce_g = self.mine_gj(gph_emb.cuda(), joint_emb.cuda())

            loss_sum_gcn, loss_sum_rel, loss_sum_att, loss_sum_img, loss_sum_all, loss_sum_all_s, loss_sum_all_s_C = 0, 0, 0, 0, 0, 0, 0

            epoch_CG += 1

            # manual batching
            np.random.shuffle(self.train_ill)
            if epoch <= 500:
                if epoch % 50 == 0:
                    k_o = 5
                    k_v = k_o - epoch // 50
                    if k_v < 1:
                        self.train_list = self.train_ill
                    else:
                        self.train_list = list_rebul_sort(self.train_ill, joint_emb, k=k_v)
                        print("K value is :" + str(k_v))
            else:
                self.train_list = self.train_ill

            if epoch % 50 == 0 and epoch <= 500 :
                print("-----------------------------------------------------------")
                print("GT sample number is :" + str(self.train_ill.shape[0]))
                print("train sample number is :" + str(self.train_list.shape[0]))
                print("-----------------------------------------------------------")


            for si in np.arange(0, self.train_list.shape[0], bsize):
                in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, att_text_emb,
                                               rel_text_emb, img_emb,
                                               self.train_list[si:si + bsize])
                align_loss = self.kl_alignment_loss(joint_emb, gph_emb, rel_emb, att_emb,
                                                    att_text_emb, rel_text_emb, img_emb, self.train_list[si:si + bsize])
                loss_all = in_loss + align_loss
                loss_all.backward(retain_graph=True)
                loss_sum_all_s = loss_sum_all_s + loss_all.item()

            loss_all = loss_all + (nce_v + nce_r + nce_g + nce_a)


            del gph_emb, rel_emb, att_emb, att_text_emb,rel_text_emb, img_emb
            torch.cuda.empty_cache()

            if epoch < 500:
                for si in np.arange(0, self.train_list.shape[0], bsize):
                    loss_joi = self.criterion_cl(joint_emb, self.train_list[si:si + bsize])
                    loss_joi.backward(retain_graph=True)
                    loss_sum_all_s = loss_sum_all_s + loss_joi.item()
                loss_all = loss_all + loss_joi

            if epoch >= 500 and epoch % 20 == 0:
                self.join_mon_emb = self.momentum_emb()
            else:
                pass

            if epoch >= 500:
                for si in np.arange(0, self.train_list.shape[0], bsize):
                    u_loss = UCL(joint_emb.to(self.device2), self.join_mon_emb, self.train_list[si:si + bsize], device=self.device2) if joint_emb is not None else 0
                    u_loss.backward(retain_graph=True)
                    loss_sum_all_s = loss_sum_all_s + u_loss.item()
                u_loss_2 = u_loss
                loss_all = loss_all + u_loss_2.detach().to(self.device)

            if epoch < 500:
                loss_all.backward()
                del loss_joi
                torch.cuda.empty_cache()

            if epoch >= 500:
                loss_all.backward()

            self.optimizer.step()

            del loss_all,joint_emb0
            torch.cuda.empty_cache()

            print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_sum_all_s, time.time() - t_epoch))
            if epoch >= self.args.il_start and (epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                # predict links
                preds_l, preds_r = self.semi_supervised_learning()

                if preds_l != -1 and preds_r != -1:
                    if (epoch + 1) % (self.args.semi_learn_step * 10) == self.args.semi_learn_step:
                        new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(preds_l)
                                     if preds_r[p] == i]
                    else:
                        new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(preds_l)
                                     if (preds_r[p] == i)
                                     and ((self.left_non_train[i], self.right_non_train[p]) in new_links)]
                    print("[epoch %d] #links in candidate set: %d" % (epoch, len(new_links)))

            if epoch >= self.args.il_start and (epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(
                    new_links) != 0 and self.args.il:

                new_links_elect = []
                for nle in new_links:
                    key = nle[0]
                    value = nle[1]
                    if key in self.dyn_link.keys():
                        v1 = self.dyn_link[key]
                        if v1 == value:
                            new_links_elect.append((key,value))
                        else:
                            self.dyn_link[key] = value
                    else:
                        self.dyn_link[key]=value

                if new_links_elect:
                    self.train_ill = np.vstack((self.train_ill, np.array(new_links_elect)))
                    print("train_ill.shape:", self.train_ill.shape)

                    num_true = len([nl for nl in new_links_elect if nl in self.test_ill_])
                    print("#true_links: %d" % num_true)
                    print("true link ratio: %.1f%%" % (100 * num_true / len(new_links_elect)))

                    with open(r"{}_true_link_note.txt".format(model_name), 'a') as f:
                        f.write('\nepoch:')
                        f.write(str(epoch))
                        f.write('\ntrue_links:')
                        f.write(str(num_true))
                        f.write("\ntrue link ratio: ")
                        ratio = (100 * num_true / len(new_links_elect))
                        f.write(str(ratio))

                    for nl in new_links_elect:
                        self.left_non_train.remove(nl[0])
                        self.right_non_train.remove(nl[1])

                    print("#entity not in train set: %d (left) %d (right)" % (
                        len(self.left_non_train), len(self.right_non_train)))


                    pred_link_ratio.append(ratio)
                    pred_link_num.append(num_true)
                    pred_wrong_num.append(len(new_links_elect)-num_true)
                    pred_epoch_list.append(epoch)

                new_links = []



            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Test
            if (epoch + 1) % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                self.test(epoch)

            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("[optimization finished!]")
        print("[total time elapsed: {:.4f} s]".format(time.time() - t_total))

    def test(self, epoch):
        with torch.no_grad():
            t_test = time.time()
            self.multimodal_encoder.eval()
            self.multi_loss_layer.eval()
            self.align_multi_loss_layer.eval()

            self.mine_gj.eval()
            self.mine_aj.eval()
            self.mine_rj.eval()
            self.mine_vj.eval()


            gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, \
            joint_emb, joint_emb0 = self.multimodal_encoder(self.device, self.input_idx,self.adj,
                                                                    self.img_features,self.rel_features,
                                                                    self.att_features,att_text_features=self.att_txt_features,
                                                                    rel_text_features = self.rel_txt_features)

            w_normalized = F.softmax(self.multimodal_encoder.fusion.weight, dim=0)
            print("normalised weights:", w_normalized.data.squeeze())

            inner_view_weight = torch.exp(-self.multi_loss_layer.log_vars)
            print("inner-view loss weights:", inner_view_weight.data)
            align_weight = torch.exp(-self.align_multi_loss_layer.log_vars)
            print("align loss weights:", align_weight.data)

            final_emb = F.normalize(joint_emb)

            # top_k = [1, 5, 10, 50, 100]
            top_k = [1, 10, 50]
            if "100" in self.args.file_dir:
                Lvec = final_emb[self.test_left].cpu().data.numpy()
                Rvec = final_emb[self.test_right].cpu().data.numpy()
                acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = multi_get_hits(Lvec, Rvec, top_k=top_k,
                                                                                        args=self.args)
                del final_emb
                gc.collect()
            else:
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                if self.args.dist == 2:  # L2
                    distance = pairwise_distances(final_emb[self.test_left], final_emb[self.test_right])
                elif self.args.dist == 1:
                    distance = torch.FloatTensor(scipy.spatial.distance.cdist(
                        final_emb[self.test_left].cpu().data.numpy(),
                        final_emb[self.test_right].cpu().data.numpy(), metric="cityblock"))
                else:
                    raise NotImplementedError

                if self.args.csls is True:
                    distance = 1 - csls_sim(1 - distance, self.args.csls_k)

                if epoch + 1 == self.args.epochs:
                    to_write = []
                    test_left_np = self.test_left.cpu().numpy()
                    test_right_np = self.test_right.cpu().numpy()
                    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

                for idx in range(self.test_left.shape[0]):
                    values, indices = torch.sort(distance[idx, :], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_l2r += (rank + 1)
                    mrr_l2r += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_l2r[i] += 1
                    if epoch + 1 == self.args.epochs:
                        indices = indices.cpu().numpy()
                        to_write.append(
                            [idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                             test_right_np[indices[1]], test_right_np[indices[2]]])
                if epoch + 1 == self.args.epochs:
                    import csv
                    save_path = self.args.save_path
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    with open(os.path.join(save_path, "pred.txt"), "w") as f:
                        wr = csv.writer(f, dialect='excel')
                        wr.writerows(to_write)

                for idx in range(self.test_right.shape[0]):
                    _, indices = torch.sort(distance[:, idx], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_r2l += (rank + 1)
                    mrr_r2l += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_r2l[i] += 1

                mean_l2r /= self.test_left.size(0)
                mean_r2l /= self.test_right.size(0)
                mrr_l2r /= self.test_left.size(0)
                mrr_r2l /= self.test_right.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
                    acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)
                del distance, gph_emb, img_emb, rel_emb, att_emb, att_text_emb, rel_text_emb, joint_emb, joint_emb0
                gc.collect()
            print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
            print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l,
                                                                                                  mean_r2l, mrr_r2l,
                                                                                                  time.time() - t_test))
            print("average: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k,
                                                                                                      (acc_l2r+ acc_r2l)/2,
                                                                                                      (mean_l2r+mean_r2l)/2,
                                                                                                      (mrr_l2r+mrr_r2l)/2,
                                                                                                  time.time() - t_test))

            ## TODO: 5_10 add
            acc_epoch_list.append(epoch)
            acc_top1 = (acc_l2r[0] + acc_r2l[0])/2
            acc_top1_list.append(acc_top1)




            with open(r"{}_accuracy_note.txt".format(model_name), 'a') as f:
                f.write('\n\nepoch:')
                f.write(str(epoch))

                f.write("\nl2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
                f.write("\nr2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k, acc_r2l,
                                                                                                  mean_r2l, mrr_r2l,
                                                                                                  time.time() - t_test))
                f.write("\naverage: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s \n".format(top_k,
                                                                                                      (acc_l2r+ acc_r2l)/2,
                                                                                                      (mean_l2r+mean_r2l)/2,
                                                                                                      (mrr_l2r+mrr_r2l)/2,
                                                                                                  time.time() - t_test))


if __name__ == "__main__":
    model = PCMEA()
    model.train()



