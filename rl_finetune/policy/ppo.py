import os
import gym
import torch
import pprint
import random 
import argparse
import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic


from ..model.net import ActorNet, CriticNet


def get_args():
    parser = argparse.ArgumentParser()
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5, 
                        help="value loss weight")
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2,
                        help="clip range, related to tianshou code : \
                        ''' surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv '''")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help="related to tianshou code : \
                        ''' nn.utils.clip_grad_norm_( \
							list(self.actor.parameters()) + list(self.critic.parameters()), \
                            max_norm=self._grad_norm) ''' ")
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1, 
                        help="whether to normalize values and returns")
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    args = parser.parse_known_args()[0]
    return args

def build_policy(args, env):
    # Model
    #net = CommonNet(device=args.device)
    actor_net = ActorNet(device=args.device,
                        dropout=args.prottrans_dropout, dropout_decay=args.prottrans_dropout_decay)
    print(args.action_shape)
    actor = Actor(actor_net, args.action_shape, device=args.device).to(args.device)
    critic_net = CriticNet(device=args.device, 
                        dropout=args.prottrans_dropout, dropout_decay=args.prottrans_dropout_decay)
    critic = Critic(critic_net, device=args.device).to(args.device)

    # Orthogonal initialization ##NOTE
    for m in list(actor.last.modules()) + list(critic.last.modules()):
        print(m)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    # Optimizer
    optim = torch.optim.Adam(
        [_ for _ in actor.parameters() if _.requires_grad]
        + [_ for _ in critic.parameters() if _.requires_grad], lr=args.lr)
    for pn, p in [_ for _ in actor.named_parameters() if _[1].requires_grad] \
        + [_ for _ in critic.named_parameters() if _[1].requires_grad]:
        print(pn, p.shape)

    dist = torch.distributions.Categorical

    # Policy
    ppo_policy = PPOPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=False,
    )

    return ppo_policy 