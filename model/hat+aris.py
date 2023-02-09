from model import common

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy
from torchvision import transforms
from torchvision.transforms import InterpolationMode
def resize_fn(img, size):
    return transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img))
import math
from model import hat


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs).double()
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x.double() * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
def make_model(args, parent=False):
    return ipt(args)

from PIL import Image
import numpy as np
def chw_to_pillow2(x):
    normalized = x.mul(255 / 255)
    tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    return Image.fromarray(tensor_cpu.numpy().astype(np.uint8))

class ipt(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ipt, self).__init__()
        
        self.scale_idx = 0
        
        self.args = args

        # new  
        n_feats = args.n_feats
        kernel_size = 3 
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        '''self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) '''
        self.hat_encoder = hat.HAT()

        self.body = VisionTransformer(img_dim=args.patch_size, 
                                      patch_dim=args.patch_dim, 
                                      num_channels=n_feats, 
                                      embedding_dim=n_feats*args.patch_dim*args.patch_dim, 
                                      num_heads=args.num_heads, num_layers=args.num_layers, 
                                      hidden_dim=n_feats*args.patch_dim*args.patch_dim*4, 
                                      num_queries = args.num_queries, 
                                      dropout_rate=args.dropout_rate, 
                                      mlp=args.no_mlp,
                                      pos_every=args.pos_every,
                                      no_pos=args.no_pos,
                                      no_norm=args.no_norm)

        self.tail = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ) for s in [2]
        ])


    def forward(self, x):
        #new
        
        if self.training:
            feature_up_scale = float(self.args.scaleh[self.scale_idx] / 2)
        else:
            feature_up_scale = float(self.args.scale[self.scale_idx] /2 )
            
        if feature_up_scale == 1:
            bicubic_img = x.clone()
            query_size = x.shape[2]
        else:
            query_size = int(round(x.shape[2] * feature_up_scale))
            if query_size % 3 !=0:
                query_size = query_size - (query_size % 3)
                
        x = self.sub_mean(x)
        x = self.hat_encoder(x)
        up_feature = torch.zeros((x.shape[0], x.shape[1], query_size, query_size)).cuda()
        t = up_feature.shape[2] / x.shape[2]

        try_up = nn.Upsample(scale_factor=t, mode='nearest')
        up_feature = try_up(x)
        
        res = self.body(x, query_size)

        res += up_feature
        x = self.hat_encoder.conv_last(self.hat_encoder.upsample(res))
        x = self.add_mean(x)

        return x 

    def set_scale(self, scale_idx):
        
        self.scale_idx = scale_idx
        
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(VisionTransformer, self).__init__()
        print(num_heads, num_layers)
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        # 64*3*3 = 576
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.img_dim = img_dim
        self.pos_every = pos_every
        # (48 / 3 = 16)^2 = 256
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        self.embed_fn, input_ch = get_embedder(int(embedding_dim/4), 0)
        self.no_pos = no_pos
        
        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.linear_encoding2 = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        self.dropout_layer2 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, query_size, con=False):
        
        patch_size = int(x.shape[-1]/3)
        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
        
        if self.mlp==False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            query_embed = None
            
        else:
            query_embed = None

        feat_coord = make_coord([patch_size, patch_size], flatten=True).cuda().unsqueeze(0)
        feat_coord = feat_coord.expand(x.shape[1], *feat_coord.shape[-2:])
        pos_embedding_fork = self.embed_fn(feat_coord).transpose(0,1).float()

        feat_coord_up = make_coord([int(query_size/3),int(query_size/3)], flatten=True).cuda().unsqueeze(0)
        feat_coord_up = feat_coord_up.expand(x.shape[1], *feat_coord_up.shape[-2:])
        pos_embedding=self.embed_fn(feat_coord_up).transpose(0,1).float()

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)
        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x+pos)
            x = self.decoder(pos_embedding, x, pos_embedding_fork)
        
        
        if self.mlp==False:
            x = self.mlp_head(x) + x
        
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            return x, con_x
        
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(query_size),self.patch_dim,stride=self.patch_dim)
        
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings
    
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory,pos_fork):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory,pos_fork)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos_fork):

        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=self.with_pos_embed(memory, pos_fork),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP_embedding(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):

        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
