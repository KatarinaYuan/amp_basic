import time 
import ipdb

import torch 
from transformers import XLNetLMHeadModel, XLNetTokenizer
import transformers 
transformers.logging.set_verbosity_error()

class BaseGenerator(object):

    def __init__(self,
        temperature=1.0, repetition_penalty=1.0, top_k=10, top_p=0.9, num_beams=1.0,
        num_sequences=5, max_seq_length=61 # </s> ##NOTE: important details 
    ):
        # Generating strategy
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p 
        self.num_beams = num_beams 
        self.num_sequences = num_sequences
        self.max_seq_length = max_seq_length

        # Get token list
        self.token_list = {}
        self.token_list['valid'] = [self.tokenizer.eos_token] + \
                            [k for k,v in self.tokenizer.get_vocab().items() if k.startswith("▁")]
        self.token_list['forbid'] = [k for k,v in self.tokenizer.get_vocab().items() if k not in self.token_list['valid']]
    
    def format_check(self, str_seqs):
        cnt_fb, cnt_ne = 0, 0 # total, with forbid_token, with no eos_token
        ret_seqs = []
        for seq in str_seqs:
            flag = True 
            for tk_forbid in self.token_list['forbid']:
                if tk_forbid in seq:
                    flag = False
                    break 
            if flag: # further check if eos_token ('</s>') only appear at the end of the seq
                eos_pos = seq.find(self.token_list[0]) 
                if eos_pos != -1 and (eos_pos + len(self.token_list[0]) == len(seq)):
                    pass # ret_seqs.append(seq)
                else:
                    cnt_ne += 1
            else: # forbidden tokens appear
                cnt_fb += 1
        if cnt_fb != 0 or cnt_ne != 0:
            print("[Warning]: format_check on generated batch:" \
                "seqs_with_forbid_token: {} seqs_without_eos: {}".format(cnt_fb, cnt_ne))
        self.num_fb = cnt_fb 
        self.num_ne = cnt_ne 
        self.num_fail = self.num_total - len(ret_seqs)
        return ret_seqs

    def generate_batch(self, dataset, gsz=1, is_check=True):
        ''' Generate a batch of sequences '''
        self.num_total = len(dataset)

        # Process prompt
        input_ids, attention_mask = dataset['input_ids'], dataset['attention_mask']

        beg_tm = time.time()
        output_ids = []
        for sz in range(0, self.num_sequences, gsz):
            ipdb.set_trace()
            o = self.model.generate(
                        input_ids=input_ids[sz: sz+gsz].to(self.device),
                        attention_mask=attention_mask[sz: sz+gsz].to(self.device),
                        max_length=self.max_seq_length, 
                        num_return_sequences=self.num_sequences,
                        temperature=self.temperature,
                        repetition_penalty=self.repetition_penalty,
                        # Sampling
                        top_k=self.top_k,
                        top_p=self.top_p,
                        do_sample=True,
                        # Beam Search 
                        num_beams=self.num_beams,
                        # num_beam_groups=self.num_beam_groups,
                        # For Debug
                        # output_scores=True,
                        # output_hidden_states=True, 
                        # return_dict_in_generate=True,
                    )
            output_ids.append(o.detach())
        output_ids = torch.cat(output_ids, dim=0)
        # e.g. o.keys(): sequences, scores, hidden_states; o.sequences.shape: (bs, 18), o.scores: tuple of length 17
        print("[generate_batch]: output_ids.shape: {} time: {}".format(output_ids.shape, time.time()-beg_tm))
        str_seqs = self.tokenizer.batch_decode(output_ids) # e.g. ['A A B</s>']
        for x in str_seqs[:5]:
            print("str_seqs: ", x)

        # Format check, in case G performs badly: e.g. generate special tokens which shouldn't be generated
        if is_check:
            str_seqs = self.format_check(str_seqs)
        
        return str_seqs  



class BackboneGenerator(BaseGenerator):

    def __init__(self, model_name_or_path, device, **kwargs):
        self.device = device 

        self.tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
        self.model = XLNetLMHeadModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

        super().__init__(**kwargs)




def get_generator(
    G_type, gen_batch_size, device, model_name_or_path='Rostlab/prot_xlnet',
    temperature=1.0, repetition_penalty=1.0, top_k=10, top_p=0.9, num_beams=5,
    num_sequences=5, max_seq_length=61 # </s> 
):
    if G_type == 'model':
        g = BackboneGenerator(model_name_or_path, device, 
                    temperature=temperature, repetition_penalty=repetition_penalty, 
                    top_k=top_k, top_p=top_p, num_beams=num_beams,
                    num_sequences=num_sequences, max_seq_length=max_seq_length)
    elif G_type == 'policy_dir':
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    return g 
