# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from component.dit_layers import *

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0,
        resid_pdrop=0,
        mlp_hidden_times=4,
        block_activate=None,
        attn_type='selfcross',
        content_spatial_size=[32,32], # H , W
        condition_dim=512,
        diffusion_step=1000,
        timestep_type='adalayernorm',
        content_emb_config=None,
        mlp_type='fc',
        context_emb = None,
        checkpoint=False,
         
    ):
        super().__init__()

        self.content_emb = context_emb

        # transformer
        assert attn_type == 'selfcross'
        all_attn_type = [attn_type] * n_layer
        
        if content_spatial_size is None:
            s = int(math.sqrt(content_seq_len))
            assert s * s == content_seq_len
            content_spatial_size = (s, s)

        self.blocks = nn.Sequential(*[Block(
               
                n_embd=n_embd,
                n_head=n_head,
               
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                attn_type=all_attn_type[n],
                content_spatial_size=content_spatial_size, # H , W
                condition_dim = condition_dim,
                diffusion_step = diffusion_step,
                timestep_type = timestep_type,
                mlp_type = mlp_type,
        ) for n in range(n_layer)])

        # final prediction head
        out_cls = self.content_emb.vocab_size-1 # num_embed: 2887
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_cls),
        )
        
       

        self.apply(self._init_weights)

  
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(self, input, cond_emb, t):
        cont_emb = self.content_emb(input) # 带有位置编码的特征
        # print('cont_emb ',cont_emb.shape)
        # assert 1==2
        emb = cont_emb

        for block_idx in range(len(self.blocks)):   
            
            emb, att_weight = self.blocks[block_idx](emb, cond_emb, t)#.cuda()) # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            
        logits = self.to_logits(emb) # B x (Ld+Lt) x n
        # print('logits ',logits.shape)
        # assert 1==2
        out = logits.transpose(-1,-2).contiguous()
        #out = rearrange(logits, 'b l c -> b c l')
        # print('out ', out.shape) #[2, 2887, 1024]
        # assert 1==2
        return out