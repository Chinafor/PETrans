import math
import logging
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertModel
import numpy as np
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICE']='1'
class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        num = int(bool(config.num_props)) + int(config.scaffold_maxlen) +1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head
        

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop = None, scaffold = None):
        b, t = idx.size()

        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.config.num_props:
            assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"           

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(torch.ones((b,t), dtype=torch.long, device=idx.device))

        pro='SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDVGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK'
        seq = [-0.035181,-0.11291,0.074672,0.014953,0.047983,-0.013022,-0.085053,0.041746,-0.054104,0.096237,0.0067627,0.044021,0.007602800000000001,0.030549,-0.0069584,0.031301,-0.0069491,-0.086995,-0.012794,-0.063341,-0.1155,0.036726,-0.065169,0.09077,-0.037119,0.0074123999999999995,0.005654600000000001,-0.020012000000000002,0.016975,0.033546,-0.021766,-0.030251,0.010653,-0.068638,-0.073507,-0.0039121,0.040789,-0.0076735,-0.053887,-0.014046000000000001,-0.070522,-0.13147999999999999,-0.031061000000000002,0.053139,0.06744299999999999,-0.00010639,-0.023803,-0.004306399999999999,0.053252,-0.033401,0.028917,-0.013850999999999999,0.034641000000000005,-0.002637,-0.012114,-0.036966000000000006,-0.10559,-0.066367,0.031412999999999996,-0.06331,0.019594999999999998,0.039832,0.051107,0.06868300000000001,0.08354400000000001,-0.013903,0.086643,-0.054657000000000004,-0.016618,0.06590700000000001,0.013034,-0.0051582,-0.10878,-0.03626,-0.06615399999999999,-0.07059299999999999,-0.0049044,0.043755,0.032131,-0.020733,0.0065946,-0.010887,-0.024054,-0.029986000000000002,-0.012736,-0.0047634999999999995,-0.080858,0.0047323,-0.11763,-0.08988099999999999,-0.0094833,-0.079694,0.018325,-0.14294,-0.019932,-0.12439000000000001,-0.16115,0.050751,1.036,0.9718200000000001,1.0113,0.96751,0.9986299999999999,1.0092,1.0396,1.1089,1.0655,0.9683,1.0628,0.9794299999999999,0.96015,0.95426,0.935,0.9138799999999999,1.0147,0.9089700000000001,1.056,1.0193,0.94101,0.98357,1.0022,1.1085,1.0341,1.0639,1.068,1.0146,0.9520299999999999,0.96325,1.0215,0.98774,1.011,1.0243,1.0412,1.0085,1.0009,1.0841,0.98887,1.1207,1.093,1.0238,1.0735,0.97651,1.1481,1.0136,1.1299,1.1653,0.96654,0,0.009584700000000002,0.0063898,0,0,0.0063898,0,0.01278,0.015974000000000002,0.0063898,0.0063898,0.0031949,0.0063898,0,0.019169,0.0031949,0.0063898,0.0031949,0.0031949,0.0031949,0,0,0.0063898,0.0031949,0.0031949,0.0031949,0,0,0.0063898,0.0063898,0.0063898,0,0,0.0063898,0,0.0063898,0.009584700000000002,0.0063898,0,0.0031949,0.0063898,0,0,0,0,0.0031949,0,0,0,0.0063898,0.009584700000000002,0.0031949,0.0031949,0.01278,0.0031949,0,0.009584700000000002,0.0063898,0.01278,0.01278,0.015974000000000002,0.0063898,0.0031949,0,0.0063898,0.01278,0.009584700000000002,0.0063898,0.0031949,0.0031949,0.0063898,0.0063898,0.009584700000000002,0,0,0.0063898,0,0.009584700000000002,0.0063898,0.0063898,0.0031949,0.0063898,0.0031949,0.0031949,0,0.0031949,0.009584700000000002,0.0031949,0.015974000000000002,0,0,0.0031949,0.0031949,0,0,0,0,0,0.0031949,0.015974000000000002,0.009584700000000002,0,0.0031949,0.009584700000000002,0,0.0031949,0.0063898,0.0063898,0.0063898,0.009584700000000002,0.009584700000000002,0.0031949,0.0063898,0.01278,0.0031949,0.0031949,0,0,0,0.0031949,0,0.0063898,0.0031949,0.0031949,0,0.0031949,0.0031949,0,0.0031949,0.0031949,0,0.0031949,0,0.0031949,0.0063898,0.0031949,0,0.0063898,0.0063898,0,0,0.0031949,0,0,0,0,0,0,0.0031949,0.0063898,0,0.0031949,0.0031949,0.0031949,0.0063898,0.009584700000000002,0.0031949,0.0031949,0,0.0031949,0,0.0063898,0.0031949,0.0031949,0,0.0031949,0.0063898,0,0,0.0063898,0.0031949,0,0,0,0,0.0031949,0.0031949,0,0,0,0,0,0.0031949,0,0,0,0.0063898,0,0,0,0,0,0,0.0031949,0,0,0.0063898,0.009584700000000002,0.009584700000000002,0.0031949,0.0031949,0,0,0.0031949,0.009584700000000002,0.0031949,0,0.009584700000000002,0.0063898,0,0.0063898,0.0063898,0,0,0,0.0031949,0,0.0063898,0.0063898,0,0,0,0.0031949,0,0,0.0031949,0,0,0,0.0031949,0,0.0063898,0.0031949,0,0,0,0.009584700000000002,0,0.0031949,0,0,0,0,0,0,0.0063898,0,0.0031949,0.0063898,0.0031949,0.009584700000000002,0,0.0031949,0.015974000000000002,0.009584700000000002,0,0,0,0,0.0031949,0.01278,0,0.0031949,0,0.0063898,0,0.0031949,0,0,0,0,0,0,0.0063898,0.01278,0,0.009584700000000002,0,0.0031949,0,0.009584700000000002,0.0063898,0.0063898,0,0,0.0063898,0,0,0,0,0,0,0,0,0,0.0031949,0,0.0031949,0,0,0,0.0031949,0.0031949,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0031949,0,0,0,0.0031949,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.049814,0.041948,0.018352,0.055057,0.015731000000000002,0.065544,0.023596000000000002,0.047192000000000005,0.020974,0.062922,0.083897,0.060301,0.031461,0.020974,0.041948,0.041948,0.034082999999999995,0.015731000000000002,0.034082999999999995,0.060301,0.084642,0.089501]
        x = np.array(seq)
        y = x.reshape(2, 256)
        pro_embeddings = torch.LongTensor(y)
        token_embeddings = token_embeddings.cuda()
        position_embeddings = position_embeddings.cuda()
        type_embeddings = type_embeddings.cuda()
        pro_embeddings = pro_embeddings.cuda()
        pro_embeddings = pro_embeddings.unsqueeze(0).cuda()
        aaa = pro_embeddings
        for i in range(85):
            aaa = torch.cat((aaa,pro_embeddings),0)


        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        x = torch.cat([aaa, x], 1)

        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))
            else:
                p = self.prop_nn(prop)
            p += type_embd
            x = torch.cat([p, x], 1)

        if self.config.scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))

            scaffold_embeds = self.tok_emb(scaffold)
            if self.config.lstm:
                scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[1][0]
                scaffold_embeds = scaffold_embeds.permute(1,0,2)
            scaffold_embeds += type_embd

            x = torch.cat([scaffold_embeds, x], 1)


        attn_maps = []
        for layer in self.blocks:

            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        if self.config.num_props and self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen) +2
        elif self.config.num_props:
            num = int(bool(self.config.num_props))
        elif self.config.scaffold:
            num = int(self.config.scaffold_maxlen) +2
        else:
            num = 0

        logits = logits[:, num:, :]


        loss = None
        if targets is not None:
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_maps