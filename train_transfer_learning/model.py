"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertModel
import numpy as np
#from proembedding import protein2emb_encoder
#from pro2 import emb
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICE']='1'
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #add num+1
        num = int(bool(config.num_props)) + int(config.scaffold_maxlen) +1 #int(config.lstm_layers)    #  int(config.scaffold)
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head
        

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # print(B)#384
        # print(T)#103
        # print(C)#256

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #aa=k.shape#384,8,103,32

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # print(att.shape)#torch.Size([384, 8, 103, 103])
        # print((self.mask[:,:,:T,:T] == 0).shape)#torch.Size([1, 1, 103, 103])
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))#float('-inf'):正无穷

        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        #print(config.n_embd2)
        #config.n_embd=256
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        # self.ln1 = nn.LayerNorm(267)
        # self.ln2 = nn.LayerNorm(267)
        # self.attn = CausalSelfAttention(config)
        # self.mlp = nn.Sequential(
        #     nn.Linear(267, 4 * 267),
        #     nn.GELU(),
        #     nn.Linear(4 * 267, 267),
        #     nn.Dropout(config.resid_pdrop),
        # )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        #print('self.tok_emb')
        #(self.tok_emb)#Embedding(94, 256)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # print('*****')
        # print(self.blocks)
        # print('*****')
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        #add
        #self.liner = nn.Linear(267, 256)
        # self.tok_emb2 = nn.Embedding(config.vocab_size, 267)
        # self.type_emb2 = nn.Embedding(2, 267)

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
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop = None, scaffold = None):
        b, t = idx.size()
        #print(b)#384
        #print(t)#53

        #print('+++++')
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.config.num_props:
            assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"           

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype=torch.long, device=idx.device))

        #add
        pro='SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDVGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK'
        #pro embeddings

        #seq = [-0.1345249516602573,0.056089747912024056,0.08144084722558513,-0.08184302517951815,0.03156727046978163,-0.05895779903678331,0.031352178383636464,-0.0723690577241808,0.1023522435414564,-0.009783772569998477,-0.033219438062283724,-0.02728130444345328,0.04897424554108798,0.009838445686677973,-0.09227964033002008,0.08392980742106508,-0.07920518479906376,0.04779959477726181,0.04710351671362844,-0.12211377608074557,0.006215167409036296,0.0460436117299616,-0.026904815708931853,0.04842913193918935,0.048661347857284716,0.020483801237573075,0.05030211486867196,-0.00982633073420318,-0.044899410554827435,-0.05061067661752865,-0.1503381022722877,0.025069523251499984,0.023490787336953516,-0.02511142166594529,0.06698313496794958,-0.018806312701745185,0.098951624540444,-0.07458466896516831,0.10027513252941213,-0.03298194017937709,-0.02101772029707228,-0.09691720495953167,0.06419025913077467,-0.015914778867550217,0.02034864697432384,0.042347298555710115,-0.0785377772824096,0.035474133681012966,-0.003789074158215122,-0.04217358618710667,-0.07834883572743169,-0.060914526662247725,-0.005212183640293842,-0.043123505964594605,0.012262497645214995,0.021017393251131448,-0.0062399805714375605,0.0005840406852486898,-0.11655441508188608,-0.0319353466597944,-0.02018558035645173,-0.02030353317639653,0.06124633339230595,-0.020768382755468568,-0.020543164761220173,-0.018368073516354118,-0.018503056405478214,-0.01879913488577841,-0.01886524071695764,-0.01856357861493119,0.06551021506871171,-0.019385866840991053,0.06587645097050418,-0.019718285661710463,-0.02011292372272066,-0.020174578029839223,-0.020360910135600586,-0.02053116373274149,-0.02080535955816424,-0.02084631286495509,-0.02089162039014457,-0.02141653621174464,-0.020766488997732774,-0.021572733279364282,-0.021664094137554376,0.06831181670519472,-0.02193737392262634,-0.02200780223087851,-0.02261305524294291,-0.022260763294199192,-0.17172926390095564,0.04021246182892056,0.05422408048190257,-0.03274004494292337,0.04595145932869446,-0.002166805489799264,0.11127767339990254,-0.1221181227607645,0.17185880962035802,-0.02173075937844111,-0.006435571783814704,-0.06981075210580077,0.06244997961380181,-0.010236826262300762,-0.04614561834762392,0.051197248905155285,-0.05545734070495515,0.010201420552787664,0.028509602062480044,-0.05059798020791408,-0.0325679809612185,0.02836803735474981,-0.027007471412407683,-0.05500900058881562,0.07754808778187099,-0.018381840899874557,-0.03344691505140795,-0.015391196390972866,-0.0933172402951266,-0.01437548204986756,0.01402626906341716,0.008850308693105312,-0.034668933399711464,0.03417231480813429,-0.050557629698335264,0.0499774735078728,-0.13590094939371591,-0.1184942801065354,-0.01355027081115536,-0.03859251032388431,-0.0009632974943732552,-0.022616863295426826,0.07027956487125196,-0.08121288100729045,0.02981677303813722,-0.05487826257700188,0.060045694656488735,0.04480517538806593,0.031342394991348665,-0.02115710583246648,-0.034614930226994516,0.03197837341184419,0.08867865416742474,0.03206284007778428,-0.0288680868859219,-0.0367854098783932,0.031230411217678708,-0.13396280611680725,0.05776851302481125,0.05745989797207233,-0.018351859075539757,0.004626307220244857,-0.06702002495686507,0.018146422117995942,-0.03263335618587798,0.04443102881667479,-0.1620047936475793,-0.10908326493936614,-0.017567148168853484,-0.03392307917981252,0.0006917821590842445,-0.011485876496917041,0.0955637379968514,-0.0827886062054329,0.014418033550184096,-0.05059389438883493,0.05438729993940352,0.07790191808895414,0.02174372089557185,-0.040808308736421085,0.023876075480028368,0.0347872864807004,0.1099790531385274,0.004330743129119178,-0.03149125837077623,-0.06349562234536063,0.03820091967240393,-0.179608150872482,0.05959495733641479,0.019304575354840262,0.01787235504126556,0.02970414064472148,-0.017143880000056875,-0.004300976328356298,-0.02997857186689192,0.09288135814426512,-0.2216273869509368,-0.11486204086574404,-0.13338238139420938,0.0006427958471909846,0.08810943942923481,-0.030517043811281255,0.08463347097435782,-0.018899788670525348,0.004516334777217143,-0.020510308762061655,0.11047734622796232,-0.06729844708880532,-0.03580536409635398,-0.02259568852354865,-0.005749342501048932,0.014987858594172346,0.1466906431644715,-0.05576678998592007,0.026504130843995367,-0.028084437873070026,0.04896270335251975,-0.03395312338897997,0.07551203322259754,-0.03795961405411743,0,0.0038911,0.0038911,0,0.019455,0.0038911,0,0.0077821,0.0038911,0.0077821,0.0077821,0.0077821,0,0,0.0077821,0.011673000000000001,0,0.0077821,0.011673000000000001,0,0,0.0077821,0.0077821,0,0.015563999999999998,0,0.0077821,0,0.0077821,0.015563999999999998,0.0038911,0.0038911,0,0.0038911,0,0.011673000000000001,0.011673000000000001,0,0,0,0,0,0,0,0,0,0,0.0038911,0,0.011673000000000001,0.011673000000000001,0,0.011673000000000001,0.0038911,0.0077821,0,0.011673000000000001,0.019455,0.0038911,0.0038911,0.011673000000000001,0.0077821,0,0.011673000000000001,0.0038911,0.011673000000000001,0.0038911,0,0.0077821,0,0.0038911,0.015563999999999998,0.0077821,0.0038911,0.0038911,0,0,0.0038911,0.011673000000000001,0.0038911,0.0038911,0.0038911,0,0,0.019455,0.0038911,0.0038911,0,0.0077821,0.0038911,0,0,0,0,0,0,0,0,0.0038911,0.0038911,0.0077821,0.015563999999999998,0.0038911,0.0038911,0.0038911,0,0.0077821,0.0038911,0,0,0.011673000000000001,0,0.015563999999999998,0.0038911,0,0.0038911,0,0,0,0,0,0.0077821,0.0038911,0.0038911,0.0038911,0,0.0077821,0.0038911,0.0038911,0,0.0038911,0,0,0,0.015563999999999998,0,0,0,0.0038911,0,0,0,0,0,0,0,0,0.0038911,0.0077821,0,0.0038911,0,0.0038911,0,0.0077821,0.0077821,0,0.011673000000000001,0.0038911,0.0038911,0,0.0038911,0.0038911,0,0,0,0.011673000000000001,0,0.0077821,0.0077821,0.0038911,0.011673000000000001,0.0038911,0.0077821,0,0.0038911,0.0038911,0,0.0038911,0,0.0038911,0,0.0038911,0.0038911,0.011673000000000001,0,0,0,0,0,0,0,0,0,0,0,0.0038911,0.0038911,0.011673000000000001,0,0.0038911,0,0,0.011673000000000001,0.0077821,0.015563999999999998,0.0038911,0.0038911,0.0038911,0,0,0,0.0038911,0.0038911,0.0038911,0,0,0,0.0038911,0,0.0077821,0.0038911,0,0,0,0.0038911,0,0,0.0038911,0.0038911,0,0.0077821,0,0,0.0038911,0,0,0,0,0,0,0,0,0,0,0.0077821,0.0038911,0.015563999999999998,0.0077821,0.0038911,0.0038911,0,0.0077821,0.011673000000000001,0.0077821,0.0077821,0,0.011673000000000001,0,0.0038911,0,0.0077821,0,0.0038911,0,0,0,0,0.0038911,0,0,0,0,0,0.0038911,0,0.0038911,0,0,0,0,0.0077821,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0038911,0,0,0,0,0,0,0,0,0,0,0,0,0.025412999999999998,0.013684,0.019549,0.037143,0.0019549,0.025412999999999998,0.021504,0.043007,0.023459,0.017594,0.050827,0.046917,0.0019549,0.021504,0.033233,0.035188,0.023459,0.013684,0.015639,0.035188,0.053413999999999996,0.048366,0.048724,0.050399,0.047106999999999996,0.049384,0.047838,0.052337,0.045993,0.050125]
        #seq = [0.0359,0.0048511000000000006,0.034956,-0.007765100000000001,-0.00020630000000000003,-0.12695,-0.0070779,0.053749,0.015512999999999999,0.011159,-0.018734,-0.065862,-0.11245999999999999,0.0048833,-0.015947,0.024887,-0.07900800000000001,0.023546,0.0014445,0.036468,0.0064158,0.022139,-0.014074000000000001,0.010359,0.014438,-0.055249,-0.13276,0.023479,0.013831,0.044994,-0.032987,0.039783,0.029994,0.028467000000000003,0.016481,0.013575,0.032764,-0.049633,0.030116000000000004,0.034891000000000005,0.040444,-0.0012344,-0.012357,0.015747,0.027287,0.0044028999999999995,-0.010399,-0.00011363,-0.025059,0.032712,0.04527,-0.028361,0.020416,-0.011557,0.004298399999999999,-0.015183000000000002,0.0038936,0.0041385,0.018300999999999998,-0.016623,0.028326999999999998,0.025488,0.017889,0.031835,-0.00048544,-0.10109,0.0084139,-0.072781,-0.066361,0.031445999999999995,-0.007855599999999999,-0.032787000000000004,0.016781,0.012495000000000001,0.021629,0.022566,0.004609899999999999,-0.00078502,-0.082919,-0.0084842,-0.058564,0.0091178,0.027884,-0.012863999999999999,-0.11829,-0.13238,0.031764999999999995,-0.1376,0.0072158000000000005,0.033983,-0.00079681,-0.007091500000000001,-0.0061247,-0.0023507,0.022318,-0.0076709000000000005,-0.01202,-0.030237,0.96811,0.95599,1.0295,0.97987,1.0128,0.997,1.0166,0.9970100000000001,0.9964200000000001,0.9828899999999999,1.0161,0.9726299999999999,0.97654,0.98498,0.9703200000000001,0.99945,1.1022,0.99041,1.0733,1.0685,0.9728700000000001,1.0106,1.0326,0.9820700000000001,0.9864799999999999,0.9774700000000001,0.9776,0.9965700000000001,1.0044,1.0816,1.0089,1.0562,0.99161,0.97391,1.0153,1.1228,1.1289,0.97034,1.1334,0.9952200000000001,0.96947,1.0045,1.0124,1.0022,1.0043,0.97265,1.0099,1.0155,1.0346,0.0017153,0.0034305,0.0051458,0.0034305,0.0017153,0,0,0.008576299999999999,0.015437000000000001,0.0051458,0.0017153,0.013722,0.0068611,0,0.017153,0.0068611,0.0034305,0.0051458,0.0051458,0,0.0034305,0,0.0051458,0.0017153,0.0034305,0.0034305,0.008576299999999999,0.0017153,0.0017153,0.0017153,0,0.008576299999999999,0.0017153,0,0,0.0017153,0.0051458,0.0017153,0.0017153,0.0017153,0.0017153,0,0,0.0017153,0.0017153,0.0017153,0,0,0,0.0034305,0.015437000000000001,0.010292,0.010292,0.0034305,0.0051458,0.0034305,0.008576299999999999,0.017153,0.010292,0.010292,0.0034305,0.0034305,0.0017153,0.0068611,0.010292,0.0068611,0.0051458,0.0034305,0.0017153,0,0.0017153,0.013722,0.0034305,0.0034305,0.0068611,0.0017153,0,0.0017153,0.0068611,0.0068611,0.0051458,0.0051458,0.0051458,0,0.008576299999999999,0,0.0034305,0.008576299999999999,0.0034305,0.0051458,0,0.0017153,0.0034305,0.0034305,0.0017153,0.0017153,0,0,0.0034305,0.012006999999999999,0.013722,0.0034305,0.0034305,0.0017153,0.0017153,0.013722,0.013722,0.0034305,0.0068611,0.0034305,0,0.0034305,0.0068611,0.0034305,0.0068611,0.0051458,0.008576299999999999,0.0034305,0.0017153,0.0068611,0.008576299999999999,0.0034305,0,0.0017153,0.0034305,0,0.0068611,0.0068611,0.0034305,0,0.0034305,0.0017153,0.0017153,0,0.0017153,0.0051458,0,0.0034305,0.0034305,0,0,0,0.0034305,0.0034305,0,0,0,0,0.0034305,0.0051458,0.0051458,0.0017153,0.0017153,0,0.013722,0.0068611,0.0034305,0.0051458,0.0034305,0.0068611,0.0034305,0.0017153,0.0017153,0.010292,0.0017153,0.0017153,0.0034305,0.0017153,0.0017153,0.0034305,0.0068611,0,0.0017153,0,0,0.0017153,0.0051458,0.0017153,0.0017153,0.0034305,0.0017153,0,0.0034305,0.0017153,0.0034305,0,0.0051458,0.0034305,0.0017153,0,0,0.0017153,0,0,0,0,0.0034305,0.008576299999999999,0.0051458,0,0.0017153,0.0017153,0,0.0034305,0.0017153,0.0068611,0.0034305,0.0034305,0.008576299999999999,0.0034305,0.0034305,0.008576299999999999,0.0051458,0.0017153,0.0034305,0,0,0.0017153,0.0051458,0.0034305,0.0051458,0,0,0,0.0017153,0.0051458,0.0068611,0,0.0017153,0.0017153,0.0017153,0.0017153,0.0017153,0.0034305,0.0017153,0,0.0051458,0,0,0,0,0.0017153,0.0017153,0,0,0.0034305,0.0068611,0.0017153,0.0017153,0.0017153,0.0034305,0,0.0034305,0,0.0034305,0.0034305,0.0017153,0.0017153,0,0.0017153,0.008576299999999999,0.0017153,0.0051458,0.0017153,0.0034305,0,0.0034305,0.0034305,0.0051458,0.0017153,0.0017153,0,0,0.0051458,0.0051458,0.0017153,0,0.0034305,0.0017153,0,0.0034305,0.0034305,0.0051458,0.0034305,0.0034305,0.0051458,0,0,0,0,0,0,0,0.0017153,0,0.0017153,0,0,0,0,0,0,0,0.0017153,0,0.0017153,0.0017153,0,0.0017153,0.0051458,0.0017153,0,0,0.0017153,0,0.0017153,0.0017153,0,0,0,0.0051458,0,0.0017153,0,0.0017153,0,0,0.0017153,0,0,0,0,0,0,0,0,0,0,0,0,0.0017153,0,0,0.042893,0.048612,0.032885000000000005,0.042893,0.027166000000000003,0.051472000000000004,0.041463,0.052901,0.024306,0.052901,0.084356,0.055761,0.024306,0.035744,0.030025,0.06576900000000001,0.040034,0.012868000000000001,0.028595,0.041463,0.08101900000000001,0.082566]
        seq = [-0.035181,-0.11291,0.074672,0.014953,0.047983,-0.013022,-0.085053,0.041746,-0.054104,0.096237,0.0067627,0.044021,0.007602800000000001,0.030549,-0.0069584,0.031301,-0.0069491,-0.086995,-0.012794,-0.063341,-0.1155,0.036726,-0.065169,0.09077,-0.037119,0.0074123999999999995,0.005654600000000001,-0.020012000000000002,0.016975,0.033546,-0.021766,-0.030251,0.010653,-0.068638,-0.073507,-0.0039121,0.040789,-0.0076735,-0.053887,-0.014046000000000001,-0.070522,-0.13147999999999999,-0.031061000000000002,0.053139,0.06744299999999999,-0.00010639,-0.023803,-0.004306399999999999,0.053252,-0.033401,0.028917,-0.013850999999999999,0.034641000000000005,-0.002637,-0.012114,-0.036966000000000006,-0.10559,-0.066367,0.031412999999999996,-0.06331,0.019594999999999998,0.039832,0.051107,0.06868300000000001,0.08354400000000001,-0.013903,0.086643,-0.054657000000000004,-0.016618,0.06590700000000001,0.013034,-0.0051582,-0.10878,-0.03626,-0.06615399999999999,-0.07059299999999999,-0.0049044,0.043755,0.032131,-0.020733,0.0065946,-0.010887,-0.024054,-0.029986000000000002,-0.012736,-0.0047634999999999995,-0.080858,0.0047323,-0.11763,-0.08988099999999999,-0.0094833,-0.079694,0.018325,-0.14294,-0.019932,-0.12439000000000001,-0.16115,0.050751,1.036,0.9718200000000001,1.0113,0.96751,0.9986299999999999,1.0092,1.0396,1.1089,1.0655,0.9683,1.0628,0.9794299999999999,0.96015,0.95426,0.935,0.9138799999999999,1.0147,0.9089700000000001,1.056,1.0193,0.94101,0.98357,1.0022,1.1085,1.0341,1.0639,1.068,1.0146,0.9520299999999999,0.96325,1.0215,0.98774,1.011,1.0243,1.0412,1.0085,1.0009,1.0841,0.98887,1.1207,1.093,1.0238,1.0735,0.97651,1.1481,1.0136,1.1299,1.1653,0.96654,0,0.009584700000000002,0.0063898,0,0,0.0063898,0,0.01278,0.015974000000000002,0.0063898,0.0063898,0.0031949,0.0063898,0,0.019169,0.0031949,0.0063898,0.0031949,0.0031949,0.0031949,0,0,0.0063898,0.0031949,0.0031949,0.0031949,0,0,0.0063898,0.0063898,0.0063898,0,0,0.0063898,0,0.0063898,0.009584700000000002,0.0063898,0,0.0031949,0.0063898,0,0,0,0,0.0031949,0,0,0,0.0063898,0.009584700000000002,0.0031949,0.0031949,0.01278,0.0031949,0,0.009584700000000002,0.0063898,0.01278,0.01278,0.015974000000000002,0.0063898,0.0031949,0,0.0063898,0.01278,0.009584700000000002,0.0063898,0.0031949,0.0031949,0.0063898,0.0063898,0.009584700000000002,0,0,0.0063898,0,0.009584700000000002,0.0063898,0.0063898,0.0031949,0.0063898,0.0031949,0.0031949,0,0.0031949,0.009584700000000002,0.0031949,0.015974000000000002,0,0,0.0031949,0.0031949,0,0,0,0,0,0.0031949,0.015974000000000002,0.009584700000000002,0,0.0031949,0.009584700000000002,0,0.0031949,0.0063898,0.0063898,0.0063898,0.009584700000000002,0.009584700000000002,0.0031949,0.0063898,0.01278,0.0031949,0.0031949,0,0,0,0.0031949,0,0.0063898,0.0031949,0.0031949,0,0.0031949,0.0031949,0,0.0031949,0.0031949,0,0.0031949,0,0.0031949,0.0063898,0.0031949,0,0.0063898,0.0063898,0,0,0.0031949,0,0,0,0,0,0,0.0031949,0.0063898,0,0.0031949,0.0031949,0.0031949,0.0063898,0.009584700000000002,0.0031949,0.0031949,0,0.0031949,0,0.0063898,0.0031949,0.0031949,0,0.0031949,0.0063898,0,0,0.0063898,0.0031949,0,0,0,0,0.0031949,0.0031949,0,0,0,0,0,0.0031949,0,0,0,0.0063898,0,0,0,0,0,0,0.0031949,0,0,0.0063898,0.009584700000000002,0.009584700000000002,0.0031949,0.0031949,0,0,0.0031949,0.009584700000000002,0.0031949,0,0.009584700000000002,0.0063898,0,0.0063898,0.0063898,0,0,0,0.0031949,0,0.0063898,0.0063898,0,0,0,0.0031949,0,0,0.0031949,0,0,0,0.0031949,0,0.0063898,0.0031949,0,0,0,0.009584700000000002,0,0.0031949,0,0,0,0,0,0,0.0063898,0,0.0031949,0.0063898,0.0031949,0.009584700000000002,0,0.0031949,0.015974000000000002,0.009584700000000002,0,0,0,0,0.0031949,0.01278,0,0.0031949,0,0.0063898,0,0.0031949,0,0,0,0,0,0,0.0063898,0.01278,0,0.009584700000000002,0,0.0031949,0,0.009584700000000002,0.0063898,0.0063898,0,0,0.0063898,0,0,0,0,0,0,0,0,0,0.0031949,0,0.0031949,0,0,0,0.0031949,0.0031949,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0031949,0,0,0,0.0031949,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.049814,0.041948,0.018352,0.055057,0.015731000000000002,0.065544,0.023596000000000002,0.047192000000000005,0.020974,0.062922,0.083897,0.060301,0.031461,0.020974,0.041948,0.041948,0.034082999999999995,0.015731000000000002,0.034082999999999995,0.060301,0.084642,0.089501]
        # embedding = nn.Embedding(53, 11)
        # embed = embedding(torch.LongTensor(seq))
        # pro_embeddings = embed.reshape(53,121)
        x = np.array(seq)
        #y = x.reshape(53,11)
        y = x.reshape(2, 256)
        pro_embeddings = torch.LongTensor(y)
        #model.to(device)
        token_embeddings = token_embeddings.cuda()
        position_embeddings = position_embeddings.cuda()
        type_embeddings = type_embeddings.cuda()
        pro_embeddings = pro_embeddings.cuda()
        pro_embeddings = pro_embeddings.unsqueeze(0).cuda()
        #
        # zero = np.zeros(53,11)
        # zero = zero.unsqueeze(0).cuda()
        #4125*384=1584000     79
        #1584079
        aaa = pro_embeddings
        #for i in range(191):
        for i in range(85):
            aaa = torch.cat((aaa,pro_embeddings),0)

        #
        #print(aaa.shape)#torch.Size([384, 53, 11])
        #print(pro_embeddings.shape)#torch.Size([1, 53, 11])
        #print((token_embeddings + position_embeddings + type_embeddings).shape)#torch.Size([384, 53, 256])
        #print(aaa.shape)
        #add 1
        #c = torch.cat(((token_embeddings + position_embeddings + type_embeddings), aaa), 1)
        #c = c.cuda()
        #print(c)
        #print(c.shape)

        ##
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        #print(x.shape)#torch.Size([384, 53, 267])
        #x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        #print(token_embeddings.shape)#torch.Size([384, 53, 256])
        #print('++++')
        #print((token_embeddings + position_embeddings + type_embeddings).shape)#torch.Size([384, 53, 256])
        #print(position_embeddings.shape)#torch.Size([1, 53, 256])
        #print(type_embeddings.shape)#torch.Size([384, 53, 256])
        #print(idx.shape)#torch.Size([384, 53])
        #print(x.shape)#384, 53, 256
        #print('idx')
        #print(idx)#[20, 20, 27,  ..., 16, 16, 16]
        #print('+++++')
        #print(aaa.shape)
        #print(x.shape)
        x = torch.cat([aaa, x], 1)
        #print('build')

        #x = self.liner(x)
        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))    # for single property
            else:
                p = self.prop_nn(prop)    # for multiproperty
            p += type_embd
            x = torch.cat([p, x], 1)
        #x = self.liner(x)

        if self.config.scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))

            scaffold_embeds = self.tok_emb(scaffold)     # .mean(1, keepdim = True)
            if self.config.lstm:
                scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[1][0]
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], 2, self.config.n_embd).mean(2)
                scaffold_embeds = scaffold_embeds.permute(1,0,2)   # mean(0, keepdim = True)
                # scaffold_embeds = scaffold_embeds.reshape(self.config.lstm_layers, 1, -1, self.config.n_embd)[-1].permute(1,0,2)
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], self.config.n_embd)
            scaffold_embeds += type_embd
            #add
            #scaffold_embeds = self.liner(scaffold_embeds)
            #print('****')
            #print(scaffold_embeds.shape)#torch.Size([384, 48, 256])
            x = torch.cat([scaffold_embeds, x], 1)


        # x = self.blocks(x)
        attn_maps = []
        #x = self.liner(x)
        for layer in self.blocks:
            # print(x)
            # print('****')
            #print(x.shape)#torch.Size([384, 53, 267])
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)
        # aaaaa = logits.shape
        # print(logits.shape)
        if self.config.num_props and self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen) +2
        elif self.config.num_props:
            num = int(bool(self.config.num_props))
        #add +2
        elif self.config.scaffold:
            num = int(self.config.scaffold_maxlen) +2
        else:
            num = 0

        logits = logits[:, num:, :]


        # if self.config.num_props or self.config.scaffold:

        #     num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)  #int(self.config.lstm_layers)   # int(self.config.scaffold)      # int(self.config.scaffold)
            

        # print(logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            #print('***')
            # a1=(logits.reshape(-1, logits.size(-1))).type
            # a11 = (logits.reshape(-1, logits.size(-1))).shape#[21120,94]
            # a2=(targets.view(-1)).type
            # a22=len(targets.view(-1))#20352 差两个bitchsize
            #print('***')
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_maps # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)