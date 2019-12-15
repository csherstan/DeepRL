#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Modified by Craig Sherstan 2019
#
######################################################################
from torch import autograd

from .BaseAgent import *
from ..network import *


class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            q_values = self._network(config.state_normalizer(self._state))['q_values']
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


def is_problem(t):
    return torch.any(torch.isinf(t)).item() or torch.any(torch.isnan(t)).item()


class TDAuxAgent(BaseAgent):
    """
    This code is based off of the DQNAgent code
    """
    def __init__(self, config):

        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

        self.losses = {}

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:

            # sampling from the replay buffer
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            # after normalizer the state is (batch, 4, 84, 84)

            next_states = self.config.state_normalizer(next_states)

            target_output = self.target_network(next_states)
            q_next = target_output["q_values"].detach()
            if self.config.double_q:
                next_network_output = self.network(next_states)["q_values"]
                best_actions = torch.argmax(next_network_output, dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            network_output = self.network(states)
            q = network_output["q_values"]
            q = q[self.batch_indices, actions]

            policy_loss = (q_next - q).pow(2).mul(0.5).mean()
            loss = policy_loss

            #---------- Start Aux tasks -----------------------#

            for n_p in self.network.named_parameters():
                if is_problem(n_p[1]):
                    raise Exception

            if is_problem(states):
                raise Exception

            self.losses.setdefault("policy", []).append(policy_loss.item())

            for key, cfg in self.network.aux_dict.items():
                # so we are applying the gamma scaling
                target = (states + cfg.gamma * target_output[key]) * (1 - cfg.gamma)
                # target = torch.zeros_like(states)
                prediction = network_output[key] * (1 - cfg.gamma)
                target.detach_()

                if is_problem(target):
                    raise Exception

                if is_problem(prediction):
                    raise Exception

                aux_loss = cfg.criteria(prediction, target)

                self.losses.setdefault(key, []).append(aux_loss.item())

                loss += aux_loss * cfg.loss_weight

            # ---------- End Aux tasks -----------------------#

            self.losses.setdefault("total_loss", []).append(loss.item())

            if self.total_steps % self.config.log_interval == 0:
                for key, loss_list in self.losses.items():
                    self.logger.add_scalar(f"loss/{key}", np.array(loss_list).mean(), step=self.total_steps)

            # TODO: there are lots of lines in here meant to debug for an instability problem observed when using
            # RMSProp.

            # copied = None
            # for n_p in self.network.named_parameters():
            #     if n_p[0] == "aux_heads.0_0.bias":
            #         copied = n_p[1].clone().detach()
            #         if is_problem(copied):
            #             raise Exception
            #
            #         break

            with autograd.detect_anomaly():
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

                # for n_p in self.network.named_parameters():
                #     if is_problem(n_p[1].grad):
                #         raise Exception
                #
                # for idx_, p in enumerate(self.optimizer.param_groups[0]["params"]):
                #     if is_problem(p):
                #         raise Exception

                with config.lock:

                    self.optimizer.step()

                    # for n_p in self.network.named_parameters():
                    #     if is_problem(n_p[1]):
                    #         raise Exception
                    #
                    # for idx_, p in enumerate(self.optimizer.param_groups[0]["params"]):
                    #     if is_problem(p):
                    #         raise Exception

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
