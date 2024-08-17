import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from component.dit import DiffusionTransformer

eps = 1e-8    #不要改这个

def sum_except_batch(x, num_dims=1): # 对num_dims 后面的维度进行求和
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a): # log(1-e_a)
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b): # M + log(e_(a-M)+e_(b-M))
    maximum = torch.max(a, b) # e(-70) 近似为0
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape): # x_shape torch.Size([2, 2888, 1024])
    # print('a ',a)
    # print('t ',t)
    # print('x_shape ',x_shape)
    b, *_ = t.shape # b,剩下的
    out = a.gather(-1, t) # 
    # print('out ',out)
    # print('(len(x_shape) - 1)) ',(len(x_shape) - 1))
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # (b,1,1)

def log_categorical(log_x_start, log_prob): # ?
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes): # 
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # 根据数值产生one-hot向量,[2, 1024, 2888]
    # print('x_onehot ', x_onehot.shape)
    permute_order = (0, -1) + tuple(range(1, len(x.size()))) # 0,-1,1
    # print('permute_order ',permute_order)
    x_onehot = x_onehot.permute(permute_order) # [2, 2888, 1024]
    # print('x_onehot ', x_onehot.shape)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30)) # 对one-hot 取log? [2, 2888, 1024]
    # print('log_x ',log_x.shape)
    return log_x

def log_onehot_to_index(log_x): # 根据log_onehot向量，找到对应的index
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.9):
    # mask and uniform
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1 # it means alpha, 等差数列
    # print('att ',att.shape)
    # print('att ',att)
    att = np.concatenate(([1], att)) # add 1 on the first
    # print('att1 ',att)
    # assert 1==2
    at = att[1:]/att[:-1] # 得到从当前步到下一步乘的系数
    # print('at ',at.shape)
    # print('at ',at)
    # assert 1==2
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1 # denotes gama,the prob for mask token
    # print('ctt ',ctt) # 与att反过来
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt # reverse
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1] # 9.99991000e-01, 9.89899091e-01
    # print('one_minus_ct ',one_minus_ct)
    ct = 1-one_minus_ct # 9.00000000e-06, 1.01009091e-02
    # print('ct ',ct)
    # assert 1==2
    bt = (1-at-ct)/N # it means beta
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    # print('att ',att)
    # print('btt ',btt)
    # print('ctt ',ctt)
    # assert 1==2
    return at, bt, ct, att, btt, ctt

class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        diffusion_step=100,
        alpha_init_type='cos',
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        mask_weight=[1,1],
        unet = None,   #this is unet we need to use
    ):
        super().__init__()
      
        self.transformer = unet  #instantiate_from_config(transformer_config) # 加载transformer
        #transformer_config['params']['content_seq_len'] # 1024  # 32 x 32 这里对长度？ 
        self.amp = False

        self.num_classes = self.transformer.content_emb.vocab_size # 1024 + 1 = 1025
       
        self.loss_type = 'vb_stochastic'
         
        self.num_timesteps = diffusion_step # 迭代的次数
        self.parametrization = 'x0' # 
        self.auxiliary_loss_weight = auxiliary_loss_weight # Reparameterization trick?
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight # [1,1] , what the means? --> the loss weight on mask region and non-mask region

        #print(self.num_timesteps,self.num_classes,"tttttcccccccc")
        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes)
            
        
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at) # 对系数求log 
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct) # log(1-e_a), log(1-ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct) # log(1-ctt)
        # M + log(e_(a-M)+e_(b-M))
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))


    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.zeros(log_x_t.size()).type_as(log_x_t)
        log_probs[:,:-1,:] = log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt)
        log_probs[:,-1:,:] = log_add_exp(log_x_t[:,-1:,:]+log_1_min_ct, log_ct)

        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1) # 保证在t在0到self.num_timesteps直接
        #根据时间步t,查找出需要的系数
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        
        # x_start: [2, 2888, 1024]
        log_probs = torch.zeros(log_x_start.size()).type_as(log_x_start) 
        log_probs[:,:-1,:] = log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt) # M + log(e_(a-M)+e_(b-M))
        log_probs[:,-1:,:] = log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)

        return log_probs

    def predict_start(self, log_x_t, cond_emb, t):          # p(x0|xt)
        # 核心是根据x_t 推理出 x0
        #print(log_x_t.shape,cond_emb.shape,"ps")
        x_t = log_onehot_to_index(log_x_t) # get the index label
        if self.amp == True:
            with autocast():
                out = self.transformer(x_t, cond_emb, t)
        else:
            # print('x_t ',x_t.shape)
            # print('cond_emb ',cond_emb.shape)
            # print('t ',t.shape)
            out = self.transformer(x_t, cond_emb, t)
        # assert 1==2
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float() # 
        batch_size = log_x_t.size()[0]
        zero_vector = torch.zeros(batch_size, 1, x_t.shape[-1]).type_as(log_x_t)- 70 # ? (log(1e-30))?
        log_pred = torch.cat((log_pred, zero_vector), dim=1) # 最后一行代表mask_token
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0)*p(x0|xt))
        # notice that log_x_t is onehot
        # log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
        #                       = log(p(x0|xt)) + log(q(xt|xt_1,x0)) + log(q(xt_1|x0)) - log(q(xt|x0))  (*)
        #                       = log(p(x0|xt)) + log(q(xt|xt_1)) + log(q(xt_1|x0)) - log(q(xt|x0))  (*)
        
        # log_x_start=log_x0_recon (the results of prediction), log_x_t=log_xt, t=t
        '''  
        
        xt 再前向t步的# q(xt+t|xt) 这个分布
        把这个分布的最后mask部分全部设为gamma
        
        计算q（xt+1｜xt） 这个分布
        应该计算这个 log(p(x0|xt)) + log(q(xt|xt_1,x0)) + log(q(xt_1|x0)) - log(q(xt|x0)) 
        log(q(x2t|xt)) 4求了
        logq(xt+1|xt) 2求了
        logx_start = log p(x0|xt)  已知也就是1
        
        q = log_x_start - log_qt    # log(p(x0|xt)/q(x2t|xt))
        norm(log(p(x0|xt)/q(x2t|xt)))
        用这个求了 第三个公式  norm(log(p(x0|xt)/q(x2t|xt)))  对这个概率前向t-1步
        self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp

        '''
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t) # get sample
        # print('log_x_t ',log_x_t.shape)
        # print('onehot_x_t ',onehot_x_t.shape)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) #选出为mask token的
        # print('mask ', mask)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t) # b,1,1 (全0)
        # print('log_one_vector ',log_one_vector)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, onehot_x_t.shape[-1]) #[2, 1, 1024]
        # log(q(xt|x0))
        log_qt = self.q_pred(log_x_t, t)  # x_t 在向前t步, 或者说把x_t 当成x_0使用？  
        #print('log_qt1 ',log_qt)                          
        log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1) # 代表mask的位置，全设为0
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape) #  # ct~
        #print('log_cumprod_ct ',log_cumprod_ct.shape)
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1) # log_x_start=log_x0_recon, b,1,1
        #print('ct_cumprod_vector ', ct_cumprod_vector.shape) # [2, 2887, 1]
        ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        # print('ct_cumprod_vector2 ', ct_cumprod_vector) # [2, 2888, 1]
        # print('log_qt ',log_qt.shape)
        # print('mask ',mask.shape) # [2, 1, 1024]
        # print((mask*ct_cumprod_vector).shape) #[2, 2888, 1024]
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector # mask token的部分，全设为ct_
        # print('log_qt2 ',log_qt)   
        # assert 1==2
        # log(q(xt|xt_1,x0))
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1), 只向前一步,因为用了 at,bt
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        q = log_x_start - log_qt    # log(p(x0|xt)/q(xt|x0))
        #print('q ',q.shape) # [2, 2888, 1024]
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        # print('q_log_sum_exp ',q_log_sum_exp.shape) # [2, 1, 1024]
        # assert 1==2 
        q = q - q_log_sum_exp       # norm(log(p(x0|xt)/q(xt|x0)))  to leverage self.q_pred
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp  # get (*), last term is re-norm
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)


    def p_pred(self, log_x, cond_emb, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, cond_emb, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, cond_emb, t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability  
        '''
         也就是根据概率采样一个xt出来
         对概率加一个随机噪声、然后从中取出最大值、也就是我们要的xt，然后对xt求onehot  log 得到新对logsample
        '''    
        uniform = torch.rand_like(logits)  #logits 是 log 概率
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30) # 产生一定的噪声 
        # print('logits ', logits)
        # print('gumbel_noise ', gumbel_noise)
        # assert 1==2
        
        sample = (gumbel_noise + logits).argmax(dim=1) # 每行最大值所在的index 
        #print('sample ',sample.shape) # [2, 1024]
        log_sample = index_to_log_onehot(sample, self.num_classes) # 又把index转为log one-hot
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        '''
            根据x0 前向t步
            首先求出logq（xt｜x0） 
            根据这个概率采样一个xt
        '''
        log_EV_qxt_x0 = self.q_pred(log_x_start, t) # 从x_0开始，往前走t步 (马尔科夫链),获得logq(xt|x0)
        # print('log_EV_qxt_x0 ',log_EV_qxt_x0.shape) # [2, 2888, 1024]
        # print('log_EV_qxt_x0 ',log_EV_qxt_x0)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0) # 根据概率分布，进行采样
        # print('log_sample ',log_sample)
        # assert 1==2
        return log_sample

    def sample_time(self, b, device, method='uniform'):
        ''' 
            取一个t  pt是取t的概率
        '''
        if method == 'importance':
            # print('self.Lt_count ',self.Lt_count)
            if not (self.Lt_count > 10).all(): # 当矩阵里每个值都大于10时，才不使用 uniform 采样
                # print('use uniform... ')
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            #print('Lt_sqrt ',Lt_sqrt)
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True) # 采index 权重大的，采到的几率就越大
            # input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
            pt = pt_all.gather(dim=0, index=t) # 根据index,找到对应的值
            # print('pt ',pt)
            # assert 1==2
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long() #从[0,num_timesteps]随机产生b个数
            # print('t ',t)
            pt = torch.ones_like(t).float() / self.num_timesteps # 概率一直都是0.01?
            # print('pt ',pt)
            # assert 1==2
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_emb, is_train=True):                       # get the KL loss
        b, device = x.size(0), x.device
        '''
            训练loss
            已知label x0
            随机取一个t
            根据t取xt
            P_theta(x0|xt) 也就是估计一个x0
            根据这个x0 计算 q(xt_1|xt,x0)  ？？？？？？？？
        '''
        assert self.loss_type == 'vb_stochastic'
        x_start = x # (b, 1024)
        t, pt = self.sample_time(b, device, 'importance') # 时间采样
        # print('pt ',pt)
        # assert 1==2

        log_x_start = index_to_log_onehot(x_start, self.num_classes) # 将数值代表，转换为由one-hot向量组成的矩阵,其中每个向量最大值所在的索引就是原始的值
        log_xt = self.q_sample(log_x_start=log_x_start, t=t) # 通过采样获得 log_xt, 随机采得
        xt = log_onehot_to_index(log_xt) # get b, 1024

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t)            # 
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob) # 用于计算x(t-1) 与 x(t)相同的数量
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]  #计算我们预测的和真实值之间一样的比例
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1] #计算xt和预测的xt-1之间相同的比例
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t) # using11111111111 true label to calculate
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        mask_region = (xt == self.num_classes-1).float() # xt 中被mask的区域
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob) # 分类的概率 e_0
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float() # t为0, p(x_0|x_1,y)
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach() # 记录下kl
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2)) # 记录加的次数,也可理解为选择了时间步t的次数

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt  # pt 代表得到采样时间的概率
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = t/self.num_timesteps + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss


    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device

    def forward(
            self, 
            x0,
            condi, 
            return_loss=True, 
            return_logits=True, 
            is_train=True,
            **kwargs):
        
        if kwargs.get('autocast') == True:
            self.amp = True
        batch_size = x0.shape[0]
        device = x0.device

        # 1) get embeddding for condition and content     prepare input
        sample_image = x0
        cond_emb = condi
    
        # cal loss and logmodelprob 
        if is_train == True:
            log_model_prob, loss = self._train_loss(sample_image, cond_emb)
            loss = loss.sum()/(sample_image.size()[0] * sample_image.size()[1])

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss 
        self.amp = False
        return out


    def sample(self,
               condition_token,
               condition_embed,
               condition_mask=None,
               content_token = None,
               filter_ratio = 0,
               temperature = 1.0,
               return_att_weight = False,
               return_logits = False,
               content_logits = None,
               print_log = True,
               **kwargs):
        
        input = {'condition_token': condition_token,
                'content_token': content_token, 
                'condition_mask': condition_mask,
                'condition_embed_token': condition_embed,
                'content_logits': content_logits,
                }

        if input['condition_token'] != None:
            batch_size = input['condition_token'].shape[0]
        else:
            batch_size = kwargs['batch_size']
    
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio) # 100*filter_ratio

        # get cont_emb and cond_emb
        if content_token != None:
            sample_image = input['content_token'].type_as(input['content_token']) # B,265

        if input.get('condition_embed_token', None) != None:
            cond_emb = input['condition_embed_token'].float()
        else:
            cond_emb = None

        if start_step == 0: # when filter_ratio==0
            # use full mask sample
            # Note that this part only support mask, mask and uniform strategies, if you use uniform strategy
            # please use sample_uniform_only() function
            #zero_logits = torch.zeros((batch_size, self.num_classes-1, cond_emb.shape[-2]),device=device) #b,256,265
            #one_logits = torch.ones((batch_size, 1, cond_emb.shape[-2]),device=device) #b,1,265
            #mask_logits = torch.cat((zero_logits, one_logits), dim=1) # 每个token全是mask
            zero_logits = torch.zeros((batch_size, self.num_classes-1, 100),device=device) #b,256,265
            one_logits = torch.ones((batch_size, 1, 100),device=device) #b,1,265
            mask_logits = torch.cat((zero_logits, one_logits), dim=1) # 每个token全是mask
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps
            #print(mask_logits.shape,"sample")
            with torch.no_grad():
                for diffusion_index in range(start_step-1, -1, -1): # 99,0
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long) #用diffusion_index填充，shape=(b,)
                    log_z = self.p_sample(log_z, cond_emb, t)     # log_z is log_onehot   

        else:
            t = torch.full((batch_size,), self.num_timesteps -1 , device=device, dtype=torch.long)
           
            sample_image = torch.randint(0, self.num_classes - 1, size=(batch_size, cond_emb.shape[-2]), device=device)
            log_x_start = index_to_log_onehot(sample_image, self.num_classes) # 
            log_xt = self.q_sample(log_x_start=log_x_start, t=t) # 向前t步
            log_z = log_xt
            with torch.no_grad():
                for diffusion_index in range(start_step-1, -1, -1): # 再依次返回
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_z = self.p_sample(log_z, cond_emb, t)     # log_z is log_onehot
        

        content_token = log_onehot_to_index(log_z) # transfer from one-hot to index
        
        output = {'content_token': content_token} # return the predict content_token
        if return_logits: # false
            output['logits'] = torch.exp(log_z)
        return output

    def sample_fast(self,
                    condition_token,
                    condition_embed,
                    condition_mask= None,
                    content_token = None,
                    filter_ratio = 0.5,
                    temperature = 1.0,
                    return_att_weight = False,
                    return_logits = False,
                    content_logits = None,
                    print_log = True,
                    skip_step = 1,
                    **kwargs):
        input = {'condition_token': condition_token,
                'content_token': content_token, 
                'condition_mask': condition_mask,
                'condition_embed_token': condition_embed,
                'content_logits': content_logits,
                }

        batch_size = input['condition_token'].shape[0]
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio)

        # get cont_emb and cond_emb
        if content_token != None:
            sample_image = input['content_token'].type_as(input['content_token'])

        if self.condition_emb is not None:
            with torch.no_grad():
                cond_emb = self.condition_emb(input['condition_token']) # B x Ld x D   #256*1024
            cond_emb = cond_emb.float()
        else: # share condition embeding with content
            cond_emb = input['condition_embed_token'].float()

        assert start_step == 0
        zero_logits = torch.zeros((batch_size, self.num_classes-1, cond_emb.shape[-2]),device=device)
        one_logits = torch.ones((batch_size, 1, cond_emb.shape[-2]),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps
        with torch.no_grad():
            # skip_step = 1
            diffusion_list = [index for index in range(start_step-1, -1, -1-skip_step)]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            # for diffusion_index in range(start_step-1, -1, -1):
            for diffusion_index in diffusion_list:
                
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = self.predict_start(log_z, cond_emb, t)
                if diffusion_index > skip_step:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-skip_step)
                else:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)

                log_z = self.log_sample_categorical(model_log_prob)

        content_token = log_onehot_to_index(log_z)
        
        output = {'content_token': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output
    
