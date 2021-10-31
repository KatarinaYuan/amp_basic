import gym
import random
import numpy as np
import torch
import math 
import ipdb

from transformers import XLNetLMHeadModel, XLNetTokenizer

from gym.spaces import Dict, Discrete, Box, Tuple
from gym import error, spaces, utils
from gym.utils import seeding

from eval_pipeline import accuracy

class SeqEnv(gym.Env):
      metadata = {'render.modes': ['human']}

      def __init__(self, args, max_seq_len, unc_factor, threshold):
            self.args = args 
            self.max_seq_len = max_seq_len + 2 # <sep>, <cls>
            self.unc_factor = unc_factor
            self.threshold = threshold

            self.tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)

            self.action_list = [k for k,v in self.tokenizer.get_vocab().items() if k.startswith("▁")]
            assert len(self.action_list) == 20
            self.action_list.append(self.tokenizer.eos_token)

            #map_token2id = self.tokenizer.get_vocab()
            #self.map_id2token = dict([(i,t) for t,i in map_token2id.items()])

            self.action_space = Discrete(len(self.action_list)) 
            self.observation_space = gym.spaces.Box(low=0, high=len(self.action_list)-1, shape=(self.max_seq_len,), dtype=np.int32)
            self.action_size = (len(self.action_list), )
            self.observation_size = (self.max_seq_len, )
            self.charstate = ""
            self.length = 0

            #ipdb.set_trace()
            #self.oracle = get_test_oracle("D2_target", "MLP") # "MLP" / "RandomForest"

            self.timer = 0
            self.mode = -1
      
      def set_reward_args(self, unc_factor, threshold):
            self.unc_factor = unc_factor
            self.threshold = threshold
            
      def set_mode(self, mode):
            self.mode = mode

      def _get_state(self):
            return self.tokenizer(self.charstate, 
                              add_special_tokens=True, 
                              pad_to_max_length=True, 
                              max_length=self.max_seq_len)["input_ids"]

      def naive_test(self):
            rew = 0
            for a in self.action_list:
                  if self.charstate.count(a) > 0:
                        rew += 1
            print("classify: ", self.charstate, rew)
            return rew / 21
      
      def step(self, action):
            self.timer += 1

            #input: action(int)
            #output: curr_state(torch.LongTensor), reward(torch.FloatTensor), done(torch.BoolTensor)
            action = self.action_list[action]
            #print("action: ", action)
            self.charstate = self.charstate + action 
            self.length += 1
            if (self.length >= self.max_seq_len) or (action == self.tokenizer.eos_token):
                  done = True
            else:
                  done = False

            if action == self.tokenizer.eos_token:
                  #o = self.oracle(self.charstate) 
                  #ipdb.set_trace()
                  o = accuracy.cal_amplike_probability(self.args, self.charstate)
                  reward = o[0, 1]
                  #oracle.evaluate_many(self.charstate)[0]

                  # Linear Combination of probability and uncertainty
                  #if o["confidence"][0][1] > self.threshold:
                  #      reward = o["confidence"][0][1] + o["entropy"][0] * self.unc_factor
                  #else:
                  #      reward = o["confidence"][0][1] - o["entropy"][0] * self.unc_factor
                  
                  # Only probability
                  #reward = o["confidence"][0][1]
                  # Naive test
                  # reward = self.naive_test()

                  # when testing
                  #if self.mode == 0:
                  #    reward = o["confidence"][0][1]
                  print("----reward: ", self.length, reward, self.charstate)
            else:
                  reward = 0
            #print("charstate: ", self.charstate, self._get_state())
            return self._get_state(), reward, done, {}

      def reset(self):
            self.charstate = ""
            self.length = 0
            return self._get_state()

      def render(self, mode='human', close=False):
            pass