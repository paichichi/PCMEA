import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------
class AdapterMLP(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        # self.n_embd = d_model if d_model is None else d_model
        # self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

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
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

        # return torch.matmul(query, key.transpose(-1, -2))

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        seq_len, _ = query.shape  # 27793

        query = self.query(query) # torch.Size([27793, 10, 10])
        key = self.key(key)  # torch.Size([27793, 10, 10])
        value = self.value(value) # torch.Size([27793, 10, 10])

        query = torch.unsqueeze(query, dim=0)
        key = torch.unsqueeze(key, dim=0)
        value = torch.unsqueeze(value, dim=0)

        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # x = torch.matmul(attn, value.transpose(-1, -2))
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        self.attn = attn.detach()
        x= torch.squeeze(x)

        x = x.reshape(seq_len, -1)

        # Output layer
        return self.output(x)

class CrossAttention(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 dropout: float):

        super().__init__()
        self.size = d_model
        self.self_attn = MultiHeadAttention(heads=10, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
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
        # self_attn = self.self_attn(query=z2, key=z2, value=z) ##original
        self_attn = self.self_attn(query=z2, key=z, value=z)  ##6-8
        x = x + self.dropout(self_attn)
        return x

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True):

        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.is_gated = is_gated

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int,
                 dropout: float):
        super(SelfAttention, self).__init__()
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.att = MultiHeadAttention(heads=10, d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        x = self.norm_self_attn(x)
        self_attn = self.att(query =x, key=x,value=x)
        x = x + self.dropout(self_attn)
        return x

# -6;-7
class CrossAndSelfAttention(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(CrossAndSelfAttention, self).__init__()
        self.adapt_mlp =  AdapterMLP(d_model, d_model//10, dropout=0.1,
                                     init_option = "lora",
                                     adapter_layernorm_option="learnable_scalar")

        self.cross_att = CrossAttention(d_model = d_model, dropout= dropout)

        self.self_att = SelfAttention(d_model=d_model, dropout=dropout)
        self.mlp = MLP(d_model= d_model, d_ff = d_model)

    def forward(self, uni_emb, cross_emb):
        fus_emb = self.cross_att(x=uni_emb, y=cross_emb)
        fus_emb = fus_emb + self.adapt_mlp(fus_emb)
        uni_emb = self.self_att(uni_emb)
        uni_emb = uni_emb + self.mlp(uni_emb)
        fus_emb = fus_emb + uni_emb
        return  fus_emb


# 11*
class Cross_And_Self_Attention_MLP_linear(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(Cross_And_Self_Attention_MLP_linear, self).__init__()
        self.cross_att = CrossAttention(d_model = d_model, dropout= dropout)

        self.self_att = SelfAttention(d_model=d_model, dropout=dropout)
        self.mlp1 = MLP(d_model= d_model, d_ff = d_model)
        self.mlp2 = MLP(d_model=d_model, d_ff=d_model)

    def forward(self, uni_emb, cross_emb):
        fus_emb = self.cross_att(x=uni_emb, y=cross_emb)
        fus_emb = fus_emb + self.mlp1(fus_emb)
        fus_emb = self.self_att(fus_emb)
        fus_emb = fus_emb + self.mlp2(uni_emb)
        return  fus_emb

# 10*
class Self_Cross_Attention_MLP_Linear(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(Self_Cross_Attention_MLP_Linear, self).__init__()
        self.cross_att = CrossAttention(d_model = d_model, dropout= dropout)

        self.self_att = SelfAttention(d_model=d_model, dropout=dropout)
        self.mlp1 = MLP(d_model= d_model, d_ff = d_model)
        self.mlp2 = MLP(d_model=d_model, d_ff=d_model)

    def forward(self, uni_emb, cross_emb):
        uni_emb = self.self_att(uni_emb)
        uni_emb = uni_emb + self.mlp1(uni_emb)
        fus_emb = self.cross_att(x=uni_emb, y=cross_emb)
        fus_emb = fus_emb + self.mlp2(fus_emb)
        return  fus_emb



class Cross_And_Self_Attention_MLP_Parallel(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(Cross_And_Self_Attention_MLP_Parallel, self).__init__()
        self.cross_att = CrossAttention(d_model = d_model, dropout= dropout)

        self.self_att = SelfAttention(d_model=d_model, dropout=dropout)
        self.mlp1 = MLP(d_model= d_model, d_ff = d_model)
        self.mlp2 = MLP(d_model=d_model, d_ff=d_model)

    def forward(self, uni_emb, cross_emb):
        fus_emb = self.cross_att(x=uni_emb, y=cross_emb)
        fus_emb = fus_emb + self.mlp1(fus_emb)
        uni_emb = self.self_att(uni_emb)
        uni_emb = uni_emb + self.mlp2(uni_emb)
        fus_emb = fus_emb + uni_emb
        return  fus_emb



class Self_att_With_Adaptor(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(Self_att_With_Adaptor, self).__init__()
        self.adapt_mlp =  AdapterMLP(d_model, d_model//10, dropout=0.1,
                                     init_option = "lora",
                                     adapter_layernorm_option="learnable_scalar")
        self.self_att = SelfAttention(d_model=d_model, dropout=dropout)


    def forward(self, emb):
        emb = emb + self.self_att(emb)
        emb = emb + self.adapt_mlp(emb)
        return emb


class Self_att_With_Mlp(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(Self_att_With_Mlp, self).__init__()
        self.self_att = SelfAttention(d_model=d_model, dropout=dropout)
        self.mlp = MLP(d_model=d_model, d_ff=d_model)

    def forward(self, emb):
        emb = emb + self.self_att(emb)
        emb = emb + self.mlp(emb)
        return emb



# 3**
class Cross_att_With_Adaptor(nn.Module):
    def __init__(self, d_model, dropout):
        super(Cross_att_With_Adaptor, self).__init__()
        self.cross_att = CrossAttention(d_model=d_model, dropout=dropout)
        self.adapt_mlp = AdapterMLP(d_model, d_model // 10, dropout=0.1,
                                    init_option="lora",
                                    adapter_layernorm_option="learnable_scalar")

    def forward(self, uni_emb, cross_emb):
        uni_emb = uni_emb + self.cross_att(x=uni_emb, y=cross_emb)
        uni_emb = uni_emb + self.adapt_mlp(uni_emb)
        return  uni_emb


#original msha
class Cross_Att_With_mlp(nn.Module):
    def __init__(self, d_model, dropout):
        super(Cross_Att_With_mlp, self).__init__()
        self.cross_att = CrossAttention(d_model=d_model, dropout=dropout)
        self.mlp = MLP(d_model=d_model, d_ff=d_model)

    def forward(self, uni_emb, cross_emb):
        uni_emb = uni_emb + self.cross_att(x=uni_emb, y=cross_emb)
        uni_emb = uni_emb + self.mlp(uni_emb)
        return  uni_emb



class LT_MSHA(nn.Module):
    def __init__(self, d_model, dropout):
        super(LT_MSHA, self).__init__()
        self.att = SelfAttention(d_model, dropout)
        self.mlp = MLP(d_model, d_model)

        self.GRU = nn.GLU(1)
        self.conv = nn.Conv1d(in_channels=d_model//2, out_channels=d_model, kernel_size=2, padding= 0, stride = 2)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, dilation=2, padding=0)
        self.FN = nn.Linear(d_model, d_model, bias=True)
        self.up = nn.ConvTranspose1d(d_model,d_model,2,2)

    def forward(self, uni_emb):
        e1 = self.att(uni_emb)
        e1 = self.mlp(e1)
        uni_emb = torch.unsqueeze(uni_emb, dim=0)
        uni_emb = uni_emb.permute(0, 2, 1)
        e2 = self.GRU(uni_emb)
        e2 = self.conv(e2)
        e2 = self.up(e2)
        e2 = e2.permute(0, 2, 1)
        e2 = torch.squeeze(e2)
        e2 = self.FN(e2)
        out = e1 + e2
        return out


class Mutual_info_Gate(nn.Module):
    def __init__(self, threhold=0.1):
        super(Mutual_info_Gate, self).__init__()
        self.threhold = threhold

    def forward(self, emb1, emb2):
        # 定义互信息阈值
        input_tensor = torch.cat([emb1, emb2], dim=1)
        input_tensor_Probs = torch.sigmoid(input_tensor)
        # target_tensor = torch.randint(0, input_tensor.shape[0], (input_tensor.shape[0],)).cuda()
        target_tensor = torch.randint(0, input_tensor.shape[0], (input_tensor.shape[0],)).cuda()
        loss = F.cross_entropy(input_tensor_Probs, target_tensor)
        mi = loss.item() - F.cross_entropy(emb1, target_tensor).item() - F.cross_entropy(emb2,target_tensor).item()

        # 记录互信息低于阈值的特征的索引
        weights = torch.ones(input_tensor.shape[1]).cuda()
        weights[mi < self.threshold] = 0.1

        # 对embedding进行加权
        weighted_emb1 = emb1 * weights[:emb1.shape[1]]
        # weighted_emb2 = emb2 * weights[emb1.shape[1]:]
        return weighted_emb1


class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.
        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """

    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        self.drop = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                out_features=x_size // 10
            )
            self.net2 = nn.Linear(
                in_features=y_size,
                out_features=x_size // 10
            )
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(self.y_size, self.x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(self.x_size, self.x_size))
            self.net = nn.Sequential(*net)

    def forward(self, x, y):
        """Calulate the score
        """
        # import ipdb;ipdb.set_trace()
        y = self.drop(y)
        x_pred = self.net(y)  # bs, emb_size

        x = self.drop2(x)
        x_2 = self.net2(x)
        del x, y

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x_2 = x_2 / x_2.norm(dim=1, keepdim=True)

        pos = torch.sum(x_2 * x_pred, dim=-1)  # bs
        torch.cuda.empty_cache()

        x_2 =  torch.matmul(x_2, x_pred.t())
        del x_pred
        # neg = torch.logsumexp(x, dim=-1)  # bs
        neg = torch.logsumexp(x_2, dim=-1)  # bs

        nce = -(pos - neg).mean()
        return nce


class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, n_class)
        # self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))

        y_2 = torch.tanh(self.linear_2(y_1))
        # y_3 = self.linear_3(y_2)
        return  y_2



def triple_loss(inputs,  targets, margin=0.3):
    ranking_loss = nn.MarginRankingLoss(margin=margin)
    n = inputs.size(0)  # batch_size

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    # dist.addmm_(1, -2, inputs, inputs.t())
    dist.addmm_(inputs, inputs.t(), beta = 1, alpha = -2)

    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    dist_ap, dist_an = [], []
    for i in range(n):
        dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    y = torch.ones_like(dist_an)
    loss = ranking_loss(dist_an, dist_ap, y)

    return loss


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar


    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()


class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)
        Args:
            T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)
        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info


def simility_cal(emb):
    sim_matric = cosine_similarity(emb, emb)
    return sim_matric

def list_rebul_sort(list, emb, k):
    sim_m = simility_cal(emb.cpu().detach().numpy())
    new_list = []
    ns_list = []
    for i in range(0, list.shape[0]):
        count = 0
        s_b = sim_m[i].argsort()
        # print(s_b)
        ns_list.append(list[i])
        for j in s_b[::-1]:
            if j==i:
                continue
            if j+1 in np.array(ns_list):
                continue
            else:
                if j+1 in list:
                    ns_list.append(list[np.where(list == (j+1))[0][0]])
                    count = count + 1
                    if count == k:
                        # print('enough')
                        break

        new_list = new_list + ns_list
        ns_list.clear()

    new_list = np.array(new_list)
    return new_list

def softXEnt(target, logits):

    logprobs = F.log_softmax(logits, dim=1)
    loss = -(target * logprobs).sum() / logits.shape[0]
    return loss

# online vs online; unsupervised
def SCL(emb, device, norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = emb, emb
    batch_size = hidden1.shape[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)
    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    return alpha * loss_a + (1 - alpha) * loss_b

# online vs target; unsupervised
def SCL2(emb, emb_m, device, norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
        emb_m = F.normalize(emb_m, dim=1)

    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = emb, emb_m
    batch_size = hidden1.shape[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)
    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    return alpha * loss_a + (1 - alpha) * loss_b

# online vs target; supervised;
def UCL(emb, target_emb, train_links, device,norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)

    zis = emb[train_links[:, 0]]
    if target_emb is not None:
        zjs = target_emb[train_links[:, 1]]

    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = zis, zjs
    batch_size = hidden1.shape[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
    # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)
    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    return alpha * loss_a + (1 - alpha) * loss_b

# online vs target; supervised; + online vs online; supervise
def S_UCL(emb, target_emb, train_links, device,norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)

    zis = emb[train_links[:, 0]]
    if target_emb is not None:
        zjs_t = target_emb[train_links[:, 1]]
    zjs_o = emb[train_links[:, 1]]

    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = zis, zjs_t
    hidden_o = zjs_o

    batch_size = hidden1.shape[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    hiddeno_large = hidden_o

    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    logits_bb_o = torch.matmul(hidden_o, torch.transpose(hiddeno_large, 0, 1)) / temperature
    logits_bb_o = logits_bb_o - masks * LARGE_NUM
    logits_abo = torch.matmul(hidden1, torch.transpose(hiddeno_large, 0, 1)) / temperature
    logits_bao = torch.matmul(hidden_o, torch.transpose(hidden1_large, 0, 1)) / temperature


    # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
    # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)

        logits_ao = torch.cat([logits_abo, logits_bb_o], dim=1)
        logits_bo = torch.cat([logits_bao, logits_aa], dim=1)
    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        logits_ao = torch.cat([logits_abo, logits_aa], dim=1)
        logits_bo = torch.cat([logits_bao, logits_bb_o], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    loss_ao = softXEnt(labels, logits_ao)
    loss_bo = softXEnt(labels, logits_bo)

    loss_out =  (alpha * loss_a + (1 - alpha) * loss_b) + (alpha * loss_ao + (1 - alpha) * loss_bo)

    return loss_out

# online vs target; supervised; + online vs target; un-supervised
def SU_UCL(emb, target_emb, train_links, device,norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)

    zis = emb[train_links[:, 0]]
    if target_emb is not None:
        zjs_t = target_emb[train_links[:, 1]]
    zjs_un = zis

    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = zis, zjs_t
    hidden_un = zjs_un

    batch_size = hidden1.shape[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    hidden_un_large = hidden_un

    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    logits_bb_o = torch.matmul(hidden_un_large, torch.transpose(hidden_un_large, 0, 1)) / temperature
    logits_bb_o = logits_bb_o - masks * LARGE_NUM
    logits_abo = torch.matmul(hidden1, torch.transpose(hidden_un_large, 0, 1)) / temperature
    logits_bao = torch.matmul(hidden_un_large, torch.transpose(hidden1_large, 0, 1)) / temperature


    # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
    # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)

        logits_ao = torch.cat([logits_abo, logits_bb_o], dim=1)
        logits_bo = torch.cat([logits_bao, logits_aa], dim=1)
    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        logits_ao = torch.cat([logits_abo, logits_aa], dim=1)
        logits_bo = torch.cat([logits_bao, logits_bb_o], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    loss_ao = softXEnt(labels, logits_ao)
    loss_bo = softXEnt(labels, logits_bo)

    loss_out =  (alpha * loss_a + (1 - alpha) * loss_b) + (alpha * loss_ao + (1 - alpha) * loss_bo)

    return loss_out

# online vs online; supervised; + online vs target; un-supervised
def SM_UCL(emb, target_emb, train_links, device,norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)

    zis = emb[train_links[:, 0]]
    # 有监督online emb
    zjs_s = emb[train_links[:, 1]]
    if target_emb is not None:
        # 无监督target emb
        zjs_t = target_emb[train_links[:, 0]]

    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = zis, zjs_s
    hidden_un = zjs_t

    batch_size = hidden1.shape[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    hidden_un_large = hidden_un

    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    logits_bb_o = torch.matmul(hidden_un_large, torch.transpose(hidden_un_large, 0, 1)) / temperature
    logits_bb_o = logits_bb_o - masks * LARGE_NUM
    logits_abo = torch.matmul(hidden1, torch.transpose(hidden_un_large, 0, 1)) / temperature
    logits_bao = torch.matmul(hidden_un_large, torch.transpose(hidden1_large, 0, 1)) / temperature


    # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
    # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)

        logits_ao = torch.cat([logits_abo, logits_bb_o], dim=1)
        logits_bo = torch.cat([logits_bao, logits_aa], dim=1)
    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        logits_ao = torch.cat([logits_abo, logits_aa], dim=1)
        logits_bo = torch.cat([logits_bao, logits_bb_o], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    loss_ao = softXEnt(labels, logits_ao)
    loss_bo = softXEnt(labels, logits_bo)

    loss_out =  (alpha * loss_a + (1 - alpha) * loss_b) + (alpha * loss_ao + (1 - alpha) * loss_bo)

    return loss_out

# online vs target; supervised; + online vs online; supervise;+ online vs target; unsupervised
def S_UCL_U(emb, target_emb, train_links, device,norm=True, inversion=False):
    if norm:
        emb = F.normalize(emb, dim=1)
        target_emb = F.normalize(target_emb, dim=1)

    zis = emb[train_links[:, 0]]
    if target_emb is not None:
        zjs_t = target_emb[train_links[:, 1]]
    zjs_o = emb[train_links[:, 1]]
    zjs_u = target_emb[train_links[:, 0]]

    temperature = 0.05
    alpha = 0.5
    n_view = 2

    LARGE_NUM = 1e9

    hidden1, hidden2 = zis, zjs_t
    hidden_o = zjs_o
    hidden_u = zjs_u

    batch_size = hidden1.shape[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    hiddeno_large = hidden_o
    hidden_u_large = hidden_u

    num_classes = batch_size * n_view
    labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
    labels = labels.to(device)

    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.to(device).float()

    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

    logits_bb_o = torch.matmul(hidden_o, torch.transpose(hiddeno_large, 0, 1)) / temperature
    logits_bb_o = logits_bb_o - masks * LARGE_NUM
    logits_abo = torch.matmul(hidden1, torch.transpose(hiddeno_large, 0, 1)) / temperature
    logits_bao = torch.matmul(hidden_o, torch.transpose(hidden1_large, 0, 1)) / temperature

    logits_bb_u = torch.matmul(hidden_u, torch.transpose(hidden_u_large, 0, 1)) / temperature
    logits_bb_u = logits_bb_u - masks * LARGE_NUM
    logits_abu = torch.matmul(hidden1, torch.transpose(hidden_u_large, 0, 1)) / temperature
    logits_bau = torch.matmul(hidden_u, torch.transpose(hidden1_large, 0, 1)) / temperature


    # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
    # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
    if inversion:
        logits_a = torch.cat([logits_ab, logits_bb], dim=1)
        logits_b = torch.cat([logits_ba, logits_aa], dim=1)

        logits_ao = torch.cat([logits_abo, logits_bb_o], dim=1)
        logits_bo = torch.cat([logits_bao, logits_aa], dim=1)

        logits_au = torch.cat([logits_abu, logits_bb_u], dim=1)
        logits_bu = torch.cat([logits_bau, logits_aa], dim=1)

    else:
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        logits_ao = torch.cat([logits_abo, logits_aa], dim=1)
        logits_bo = torch.cat([logits_bao, logits_bb_o], dim=1)

        logits_au = torch.cat([logits_abu, logits_aa], dim=1)
        logits_bu = torch.cat([logits_bau, logits_bb_u], dim=1)

    loss_a = softXEnt(labels, logits_a)
    loss_b = softXEnt(labels, logits_b)

    loss_ao = softXEnt(labels, logits_ao)
    loss_bo = softXEnt(labels, logits_bo)

    loss_au = softXEnt(labels, logits_au)
    loss_bu = softXEnt(labels, logits_bu)

    loss_out =  (alpha * loss_a + (1 - alpha) * loss_b) + (alpha * loss_ao + (1 - alpha) * loss_bo) + (alpha * loss_au + (1 - alpha) * loss_bu)

    return loss_out




