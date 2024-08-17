import torch
import torch.nn as nn
from component.dit import DiffusionTransformer
from component.discretediffusion import DiscreteDiffusion,index_to_log_onehot,log_onehot_to_index
import dac
import torchaudio
from component.embedding import BERTEmbedding
import torch.nn.functional as F

from modeling_pretrain import EmbeddingModel


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=62, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=9, stride=5, padding=2)
        self.resblock1 = ResidualBlock(512)
        self.resblock2 = ResidualBlock(512)
        self.resblock3 = ResidualBlock(512)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # (b, 62, 1000) -> (b, 128, 500)
        x = F.relu(self.conv2(x))  # (b, 128, 500) -> (b, 512, 100)
        x = self.resblock1(x)      # (b, 512, 100) -> (b, 512, 100)
        x = self.resblock2(x)      # (b, 512, 100) -> (b, 512, 100)
        x = self.resblock3(x)      # (b, 512, 100) -> (b, 512, 100)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.truncation_forward = False
        codec_model_path = dac.utils.download(model_type="44khz")
        self.codec = dac.DAC.load(codec_model_path)

        self.codec.eval()
 
 
        tokenmax = 1024 
        modeldim = 512
        n_layer = 20
        n_head = 16
        
        
        self.embedding = EmbeddingModel()
        
       
        context_emb = ChunkedEmbedding(codebook_size=tokenmax, n_quantizer=32, dim=modeldim)
        
        
        Dit = DiffusionTransformer(
                                    n_layer=n_layer,
                                    n_embd=modeldim,
                                    n_head=n_head,
                                   
                                    attn_pdrop=0,
                                    resid_pdrop=0,
                                    mlp_hidden_times=4,
                                    block_activate='GELU2',
                                    attn_type='selfcross',
                                    
                                    condition_dim=modeldim,
                                    diffusion_step=100,
                                    timestep_type='adalayernorm',
                                    mlp_type='fc',
                                    context_emb = context_emb)
        
        self.vqdiffusion = DiscreteDiffusion(diffusion_step=100,
                                             alpha_init_type='alpha1',
                                             auxiliary_loss_weight=5.0e-4,
                                             adaptive_auxiliary_loss=True,
                                             mask_weight=[1,1],
                                             unet = Dit)
        
        self.eegproj = nn.Linear(200,modeldim)
    
    
    
    @torch.no_grad()
    def getembedding(self, eeg):
        emb =self.embedding(eeg,63,None).unsqueeze(1).contiguous() 
         
        return emb
    
    # wav经过codec 得到token
    @torch.no_grad()
    def encode(self, wav):
        
        token = self.codec.encode(wav, bw=-1).squeeze(-1)
        return token
    
    # token转wav 
    @torch.no_grad()
    def decode(self, token):
        audio_values = self.codec(token.unsqueeze(-1), bw=-1)
        return audio_values

    
    def forward(self, eeg, audio): 
        #经过codec得到低采样率的token
        #code_l = self.encode(wav_l) # b tokennum
        audio_code = self.encode(audio)    
        #print(eeg.shape) 
        eeg = eeg.view(eeg.shape[0],62,-1,200)
        #print(eeg.shape)
        eeg = getembedding(eeg)#self.embedding(eeg,63,None).unsqueeze(1).contiguous() #.transpose(-1,-2).contiguous() 
        eeg  = self.eegproj(eeg)
        #lr_emb = self.embdding(code_l)  #b tokennum 1024
        
        pred_s = self.vqdiffusion(x0 = audio_code, condi = eeg)   #inpt hr [b 200]  lr emb[b, 200 1024]
        return pred_s['loss']

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper

    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True) 
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:,0:1,:], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:,:-1,:]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs
            return wrapper

        else:
            print("wrong sample type")
            
   

    
    @torch.no_grad()
    def generate_content(self,
                         *,
        		 batch,
        		 filter_ratio = 0,
        		 temperature = 1.0,
        		 content_ratio =1,
        		 replicate=1,
        		 return_att_weight=False,
        		 sample_type="top0.85r"):
        
        self.eval()
        
        eeg = batch#self.encode(batch)
        eeg = eeg.view(eeg.shape[0],62,-1,200)
        #print(eeg.shape)
        eeg = getembedding(eeg)#self.embedding(eeg,63,None).unsqueeze(1).contiguous() #.transpose(-1,-2).contiguous() 
        eeg  = self.eegproj(eeg)
        #eeg = self.embedding(eeg).transpose(-1,-2) #b 100 512
        batchsize = eeg.size(0)
        
        #eeg = codelr.view(batchsize,-1)
        #codelremb  = self.embdding(codelr)
        condition = {}
        condition['condition_token'] = eeg
        condition['condition_embed_token'] = eeg
 
        if replicate != 1: 
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
         
        
        # assert 1==2
        content_token = None

        if len(sample_type.split(',')) > 1: # using r,fast
            if sample_type.split(',')[1][:1]=='q':
                self.decoder.p_sample = self.p_sample_with_truncation(self.decoder.p_sample, sample_type.split(',')[1])
        
        
        #我们只用了下面这个predict_start_with_truncation + self.vqdiffusion.sample 这样采样100步的方式
        
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:              
            self.vqdiffusion.predict_start = self.predict_start_with_truncation(self.vqdiffusion.predict_start, sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='fast':
            trans_out = self.vqdiffusion.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition['condition_embed_token'],
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(sample_type.split(',')[1][4:]))

        else:
            trans_out = self.vqdiffusion.sample(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition['condition_embed_token'],
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type)
        
        
        
        decoderout = trans_out['content_token']    #这里得到的是预测的code
        
        
         
        predicted_wav = self.decode(decoderout)  #经过codec的解码器得到wav输出
        
     
        self.train()
        out = {'content': predicted_wav}
    
        return out
    
if __name__ == '__main__':
    # define model
    model = Generator()
    # define sig
    wav = torchaudio.load('p374_424.wav')[0]
    wav = torchaudio.transforms.Resample(orig_freq=48000,new_freq=24000)(wav)
    wav = wav[:,:24000].unsqueeze(1)   #[1 1 t]
    
    print(f'wav.shape  {wav.shape}')
 
    eeg = torch.randn(1,62,1000)
    pred_s = model(eeg, wav)
    print(pred_s)
    
    sample = model.generate_content(batch = eeg,replicate=1)
    print(sample['content'].shape)
    '''
    from thop import profile
    macs, params = profile(model, inputs = (features,target_features))
    print('MACs:', macs/2**30, '[G] Params:', params/2**10,'[k × 4 Bytes]')
    '''

