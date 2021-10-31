import os
import gym
import torch
import pprint
import random 
import argparse
import datetime
import ipdb
from argparse import Namespace
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer

from gym.envs.registration import register

from .policy import ppo 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default="seq-env-v0")
    parser.add_argument('--policy', type=str, default="ppo", choices=["ppo"])

    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--epoch', type=int, default=300) ##
    parser.add_argument('--step-per-epoch', type=int, default=300) ##
    parser.add_argument('--episode-per-collect', type=int, default=10, ##
                        help="how many amp sequences would be generated")
    parser.add_argument('--repeat-per-collect', type=int, default=3, ##
                        help="the number of repeat time for policy learning, \
							for example, set it to 2 means the policy needs \
							to learn each given batch data twice.")
    parser.add_argument('--episode-per-test', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[16, 16])
    parser.add_argument('--training-num', type=int, default=20) ##
    parser.add_argument('--test-num', type=int, default=5) ##
    parser.add_argument('--render', type=float, default=0.)
    
    parser.add_argument('--max-seq-len', type=int, default=60) # NOTE
    parser.add_argument('--rewfn-unc-factor', type=float, default=0.)
    parser.add_argument('--rewfn-threshold', type=float, default=0.8)
    parser.add_argument('--stopcnt-threshold', type=int, default=5)
    parser.add_argument('--gen-batch-size', type=int, default=100)
    parser.add_argument('--prottrans-dropout', type=float, default=0.5)
    parser.add_argument('--prottrans-dropout-decay', type=int, default=1000)
    parser.add_argument('--ent-threshold', type=float, default=0.6)

    args_train = parser.parse_known_args()[0]
    if args_train.policy == "ppo":
        args_policy = ppo.get_args()
    else:
        raise NotImplementedError
    args = Namespace(**vars(args_train), **vars(args_policy))
    return args 

def policy_tester(args, policy):
    env = gym.make(args.task)
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=args.test_num, render=args.render)
    rews, lens = result["rews"], result["lens"]
    return rews.mean(), lens.mean()

def run_onpolicy(args, env, train_envs, test_envs, policy):

    # Collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # Log
    log_path = f'seed_{args.seed}_{args.t0}-{args.task.replace("-", "_")}_{args.policy}' + args.userinfo
    log_path = os.path.join(args.logdir, args.task, args.policy, log_path)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer, update_interval=100, train_interval=100)

    def save_fn(policy):
        #ipdb.set_trace()
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    stop_cnt = 0
    def stop_fn(mean_rewards):
        # entropy
        """test_collector.reset_env()
        test_collector.reset_buffer()
        policy.eval()
        result = test_collector.collect(n_episode=20)
        batch, indice = test_collector.buffer.sample(0)
        dist = policy(batch).dist
        ent = dist.entropy().mean()
        print("-----------------------stop_func: ", ent)"""
        nonlocal stop_cnt
        if mean_rewards >= env.spec.reward_threshold:
            stop_cnt += 1
        return stop_cnt > args.stopcnt_threshold #or ent < args.ent_threshold

    def test_fn(epoch, env_step):
        pass 
        '''if epoch % 30 == 0:
            ckpt_path = os.path.join(log_path, 'policy_eval.pth')
            torch.save(policy.state_dict(), ckpt_path)
            '''

    # Trainer
    #with torch.autograd.detect_anomaly():
    result = onpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.repeat_per_collect, args.episode_per_test, args.batch_size,
            episode_per_collect=args.episode_per_collect, stop_fn=stop_fn, save_fn=save_fn,
            test_fn=test_fn,
            logger=logger)
    #assert stop_fn(result['best_reward'])
    pprint.pprint(result)

    torch.save(policy.state_dict(), os.path.join(log_path, 'last_policy.pth'))

    if __name__ == '__main__':
        # Let's watch its performance!
        rews, lens = policy_tester(args, policy)
        print(f"Final reward: {rews}, length: {lens}")

def register_gym_game(args):
    """
    To use custom env, run `pip install -e .` under './custenv'
    """
    register(
        id='seq-env-v0',
        entry_point='rl_finetune.custenv.custenv.envs:SeqEnv',
        kwargs={'args':args, 'max_seq_len': args.max_seq_len, 'unc_factor': args.rewfn_unc_factor, 'threshold': args.rewfn_threshold},
        reward_threshold=1000,
        max_episode_steps=100
    )
    env = gym.make(args.task)
    args.state_shape = env.observation_size
    args.action_shape = env.action_size #NOTE
    print(args.state_shape, args.action_shape)

    return env 

def train(args):

    # Register gym game
    env = register_gym_game(args)
    # Build envs 
    train_envs = DummyVectorEnv( # SubprocVectorEnv
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])

    if args.policy == "ppo":
        # Policy
        policy = ppo.build_policy(args, env)
        run_onpolicy(args, env, train_envs, test_envs, policy)
    else:
        raise NotImplementedError