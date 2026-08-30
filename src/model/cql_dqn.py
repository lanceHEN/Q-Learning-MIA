import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class CQLDQN(DQN):
    """
    Conservative Q-Learning (CQL) for discrete action spaces, built on top of
    SB3's DQN. Adds a penalty term to the TD loss that pushes down Q-values
    for actions not present in the replay buffer, mitigating the OOD
    over-estimation problem in offline RL.

    CQL penalty (discrete):
        L_CQL = logsumexp_a Q(s, a) - Q(s, a_data)

    Total loss:
        L = L_TD + cql_alpha * L_CQL

    With cql_alpha=0 this reduces to standard DQN.
    """

    def __init__(self, *args, cql_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cql_alpha = cql_alpha

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # All Q-values for the current observations: [batch, n_actions]
            all_q_values = self.q_net(replay_data.observations)

            # Q-value for the action taken (in the dataset): [batch, 1]
            current_q_values = th.gather(all_q_values, dim=1, index=replay_data.actions.long())

            # TD loss
            td_loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # CQL penalty: penalise Q-values for all actions relative to the
            # dataset action, discouraging over-estimation of unseen actions.
            cql_loss = (th.logsumexp(all_q_values, dim=1) - current_q_values.squeeze(1)).mean()

            loss = td_loss + self.cql_alpha * cql_loss
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
