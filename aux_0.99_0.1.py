import argparse
import os

import torch

from deep_rl import random_seed, set_one_thread, select_device, Config, generate_tag, Task, TDAuxNet, NatureConvBody, \
    LinearSchedule, AsyncReplay, ImageNormalizer, SignNormalizer, run_steps, mkdir
from deep_rl.agent.TDAux_agent import TDAuxAgent


def td_aux_many(config: Config, **kwargs):
    """

    :param config:
    :param kwargs: kwargs used to generate the experiment tag name uses for saving.
    :return:
    """
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config.merge(kwargs)

    mkdir(os.path.join(config.data_dir, 'log'))
    mkdir(os.path.join(config.data_dir, 'data'))

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    # aux_gammas = [0.0, 0.5, 0.9, 0.99]
    aux_gammas = [0.99]

    aux_dict = {str(g).replace(".", "_"): TDAuxNet.AuxCFG(g, loss_weight=0.1) for g in aux_gammas}
    # aux_dict = {}

    # config.optimizer_fn = lambda params: torch.optim.RMSprop(
    #     params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4)
    # I'm just hard coding the shape of the target
    config.network_fn = lambda: TDAuxNet((84, 84), config.action_dim,
                                         NatureConvBody(in_channels=config.history_length), aux_dict)
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.replay_fn = lambda: AsyncReplay(memory_size=int(5e5), batch_size=32)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.history_length = 4
    # config.double_q = True
    config.double_q = False
    run_steps(TDAuxAgent(config))


if __name__ == "__main__":
    cf = Config()
    cf.add_argument('--game', required=True)
    cf.add_argument('--run', type=int, required=True)
    cf.add_argument('--data_dir', type=str, required=True)
    cf.add_argument('--save_interval', type=int, default=1000000)
    cf.add_argument('--max_steps', type=int, default=int(2.5e7))
    cf.merge()

    set_one_thread()
    select_device(0)

    td_aux_many(cf, game=cf.game, run=cf.run, remark="aux_0.99_0.1")
