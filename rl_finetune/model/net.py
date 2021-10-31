import torch
import torch.nn as nn 
from transformers import XLNetLMHeadModel, XLNetTokenizer

import transformers 
transformers.logging.set_verbosity_error()

class ProttransNet(nn.Module):

    tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
    model = XLNetLMHeadModel.from_pretrained("Rostlab/prot_xlnet")

    def __init__(self, device, dropout, dropout_decay):
        super().__init__()
        self.device = device 
        
        #self.tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
        #self.model = XLNetLMHeadModel.from_pretrained("Rostlab/prot_xlnet").to(device)
        
        ProttransNet.model = ProttransNet.model.to(device)
        ProttransNet.model.requires_grad_(False)
        ProttransNet.model.eval() ##NOTE
    
    """@classmethod
    def set_dropout_rate(cls):
        if cls.step % cls.dropout_decay_intvl == 0:
            r = cls.dropout * \
                    max(0, (1 - (cls.step // cls.dropout_decay_intvl) / cls.dropout_decay_step))
            print("[ProttransNet.set_dropout_rate]: dropout: ", r)
            for m in cls.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = r
            cls.cur_dropout = r """

    def process(self, obs, should_train):
        ProttransNet.model.train(should_train) #(False)

        ## obs: np.ndarray:: (bsz, MAX_LEN+2) # <sep>, <cls>
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.int32)
        input_mask = (obs == ProttransNet.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)).int()
        batch_size = obs.shape[0]
        with torch.no_grad():
            res = ProttransNet.model(obs, input_mask=input_mask, output_hidden_states=True)
            # res.logits: (bsz, sl, vocab_size)
            # res.mems: tuple of length 30 (30 layers), res.mems[0]: (sl, bsz, d_model)
            # res.hidden_states: tuple of length 31 (1 embedding layer, 30 layers), res.hidden_states[0]: (bsz, sl, d_model)

            # Get feature vector
            feat = res.hidden_states[-1].sum(dim=1) # (bsz, d_model)
            #feat = torch.stack(res.hidden_states, dim=0).sum(dim=0).sum(dim=1) # (bsz, d_model)
            #feat = res.logits[:, -1, :]
            logits_old = res.logits[:, -1, :] # (bsz, vocab_size)

        #print("prottrans.obs:", obs)
        #print("prottrans.feat:", feat, logits_old)

        return feat, logits_old

class ActorNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = ProttransNet(**kwargs)

        logits_old_dim = self.encoder.model.config.vocab_size
        self.output_dim = logits_old_dim
        self.logits_old_norm = nn.LayerNorm(logits_old_dim)

    
    def forward(self, obs, state=None, info={}):
        _, logits_old = self.encoder.process(obs, self.training)
        # Normalize 
        logits_old = self.logits_old_norm(logits_old)

        #print("----->ActorNet.logits:  ", logits_old, logits_old[:, 5])
        
        return logits_old, state 

class CriticNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder = ProttransNet(**kwargs)

        feat_dim = self.encoder.model.config.d_model
        logits_old_dim = self.encoder.model.config.vocab_size
        self.output_dim = feat_dim + logits_old_dim # cat
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.logits_old_norm = nn.LayerNorm(logits_old_dim)

    
    def forward(self, obs, state=None, info={}):
        feat, logits_old = self.encoder.process(obs, self.training)
        # Normalize 
        feat = self.feat_norm(feat)
        logits_old = self.logits_old_norm(logits_old)

        return torch.cat((feat, logits_old), dim=1), state