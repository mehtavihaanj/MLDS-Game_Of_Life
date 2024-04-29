from .networks import Actor, Critic
from battle.envs import ParallelGridBattleRL
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import math


class PPO:
    def __init__(self, env: ParallelGridBattleRL,
                 max_episode_timesteps: int = 50,
                 actor: str = None,
                 critic: str = None):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.agent_id = 0  # play as agents 0
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.actor = Actor(self.env)
        if actor:
            actor_file = torch.load(actor)
            self.actor.load_state_dict(actor_file['state_dict'])
            self.best_actor_loss = actor_file['loss']
        else:
            self.best_actor_loss = np.inf

        self.critic = Critic(self.env)
        if actor:
            critic_file = torch.load(critic)
            self.critic.load_state_dict(critic_file['state_dict'])
            self.best_critic_loss = critic_file['loss']
        else:
            self.best_critic_loss = np.inf

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        # -----------hyperparams-----------
        self.gamma = 0.95  # discount factor for calculating rewards-to-go
        self.max_episode_timesteps = max_episode_timesteps  # max timesteps per episode
        self.epochs_per_iter = 5  # number of update epochs per iteration
        self.clip = 0.2  # as recommended by the paper
        self.lr = 0.005  # learning rate for networks, u can make separate ones for actor and critic if u want
        # ---------------------------------

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)  # optimizer for actor network
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)  # optimizer for critic network

    def learn(self, iterations: int) -> None:
        print(f'Learning for {iterations} iterations')
        for iteration in range(iterations):
            print(f'Iteration {iteration}')

            # ALG STEP 3
            batch_obs, batch_mask, batch_action, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_action)

            # ALG STEP 5
            # Calculate overall advantage at this epoch
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for epoch in range(self.epochs_per_iter):
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, cur_n_log_probs = self.evaluate(batch_obs, batch_action)

                # Calculate ratios
                ratios = torch.exp(cur_n_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr_1 = ratios * A_k
                surr_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr_1[batch_mask], surr_2[batch_mask])).mean()  # only consider valid states

                if actor_loss.item() < self.best_actor_loss:
                    self.best_actor_loss = actor_loss.item()
                    torch.save({
                        'state_dict': self.actor.state_dict(),
                        'loss': actor_loss.item(),
                    }, 'actor.pth')

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)  # retain it for later when we optimize critic
                self.actor_optim.step()

                critic_loss_criterion = nn.MSELoss()
                critic_loss = critic_loss_criterion(V[batch_mask], batch_rtgs[batch_mask])

                if critic_loss.item() < self.best_critic_loss:
                    self.best_critic_loss = critic_loss.item()
                    torch.save({
                        'state_dict': self.critic.state_dict(),
                        'loss': critic_loss.item(),
                    }, 'critic.pth')

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                print(f'epoch {epoch:3} | '
                      f'actor loss: {actor_loss.item():2.9f} | '
                      f'critic loss: {critic_loss.item():2.9f}')


    def evaluate(self, batch_obs, batch_action):
        output_obs_shape = batch_obs.shape  # (T, N, C, H, W)

        # Turns (T, N, C, H, W) batch obs to (T*N, C, H, W) for processing
        batch_obs_flattened = batch_obs.view(-1, *batch_obs.shape[2:])
        # Turns (T, N, H, W) batch action to (T*N, H*W) for taking distributions
        batch_action_flattened = batch_action.view((batch_obs_flattened.shape[0], -1))

        # Query critic network for a value V for each obs for each timestep
        # Changes to (T, N) after processing
        V = self.critic(batch_obs_flattened).view(output_obs_shape[:2])

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_n_action()
        cur_n_action_logits = self.actor(batch_obs_flattened)

        # Sample an attack location from the distribution and get its log prob
        flattened_cur_n_action_logits = cur_n_action_logits.view((cur_n_action_logits.shape[0], -1))
        cur_n_dist = torch.distributions.OneHotCategorical(logits=flattened_cur_n_action_logits)

        cur_n_log_probs = cur_n_dist.log_prob(batch_action_flattened).view(output_obs_shape[:2]).to(self.device)

        return V, cur_n_log_probs

    def rollout(self):
        default_action = np.zeros(shape=self.env.attack_window)
        default_action[2, 2] = 1
        # batch data
        batch_mask = np.empty(shape=(0, math.prod(self.env.shape)))  # mask tells which observations are actually valid
        batch_obs = np.empty(shape=(0, *self.obs_shape))  # (timestep, N, C, H, W)
        batch_action = np.empty(shape=(0, *self.act_shape))
        batch_log_probs = np.empty_like(batch_mask)
        batch_rewards = np.empty(shape=(0, self.obs_shape[0]))  # (timestep, episode (N))
        batch_rtgs = np.empty_like(batch_rewards)
        batch_lens = np.empty(shape=(self.obs_shape[0]))

        n_obs, masks = self.env.reset()  # this is different from batch masks, which only mask for validity
        ep_t = 0

        for ep_t in range(self.max_episode_timesteps):
            # Collect observation
            batch_obs = np.append(batch_obs, np.expand_dims(n_obs, 0), axis=0)

            # Verify that observation is valid/invalid (by checking if max is 0) and mark it in the mask
            # batch_mask = np.append(batch_mask, self.produce_mask(n_obs), axis=0)

            # Only use observations for our agents
            batch_mask = np.append(batch_mask, np.expand_dims(masks[self.agent_id], axis=0), axis=0)

            # Transfer obs to cuda
            n_obs = torch.from_numpy(n_obs).float().to(self.device)

            n_action, n_log_probs = self.get_n_action(n_obs)

            # -- train against default action bots
            n_action[masks[self.agent_id] == False] = default_action

            # -- train against bots that attack whenever they can
            # enemy_obs = n_obs[masks[1 - self.agent_id]]
            # enemy_action = torch.empty(size=(enemy_obs.shape[0], *self.act_shape[1:])).to(self.device)
            # enemy_valid_actions = enemy_obs[:, 0, ...] + enemy_obs[:, 3, ...]
            # enemy_valid_actions_sum = enemy_obs[:, 0, ...].sum(dim=(1, 2)) + enemy_obs[:, 3, ...].sum(dim=(1, 2))
            # enemy_valid_actions_mask = enemy_valid_actions_sum > 0
            # enemy_random_actions_mask = enemy_valid_actions_mask == False
            # enemy_action[enemy_valid_actions_mask] = enemy_valid_actions[enemy_valid_actions_mask] / enemy_valid_actions_sum[enemy_valid_actions_mask].unsqueeze(-1).unsqueeze(-1).expand(-1, *self.act_shape[1:])
            # if enemy_random_actions_mask.sum() > 0:
            #     enemy_action[enemy_random_actions_mask] = enemy_obs[enemy_random_actions_mask][:, 2, ...] / enemy_obs[enemy_random_actions_mask][:, 2, ...].sum(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            # flattened_enemy_action = enemy_action.flatten(start_dim=1)
            #
            # dist = torch.distributions.OneHotCategorical(probs=flattened_enemy_action)
            # action_flat = dist.sample()
            # enemy_action = action_flat.view(size=(enemy_obs.shape[0], *self.act_shape[1:]))
            #
            # n_action[masks[1 - self.agent_id]] = enemy_action.detach().cpu().numpy()

            n_full_obs, n_reward, done, _ = self.env.step(n_action)
            n_obs, masks = n_full_obs

            # Collect action, reward, and log prob
            batch_rewards = np.append(batch_rewards, np.expand_dims(n_reward, 0), axis=0)
            batch_action = np.append(batch_action, np.expand_dims(n_action, 0), axis=0)
            batch_log_probs = np.append(batch_log_probs, np.expand_dims(n_log_probs, 0), axis=0)

            if done:
                break

        batch_lens = np.full_like(batch_lens, ep_t + 1)

        # Reshape data as tensors in the shape specified before returning
        batch_mask = torch.tensor(batch_mask, dtype=torch.bool, device=self.device)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rewards)
        # Return the batch data
        return batch_obs, batch_mask, batch_action, batch_log_probs, batch_rtgs, batch_lens

    # from an n_obs ndarray, return an array of shape (N) of which observations are valid (have a max > 0)
    @staticmethod
    def produce_mask(n_obs):
        return np.amax(n_obs, axis=(1, 2, 3)).reshape((1, n_obs.shape[0])) < 1

    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (# timesteps, # episodes)
        batch_rtgs = np.empty_like(batch_rewards)

        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for j in reversed(range(batch_rewards.shape[1])):  # iterate thru episodes
            episode_rewards = batch_rewards[:, j, ...]
            discounted_reward = 0  # The discounted reward so far
            for i, reward in reversed(list(enumerate(episode_rewards))):  # iterate thru timesteps
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs[i, j] = discounted_reward

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
        return batch_rtgs

    def get_n_action(self, n_obs):
        # Query the actor network for attack distributions
        n_action_logits = self.actor(n_obs)

        # Sample an attack location from the distribution and get its log prob
        flattened_n_action_logits = n_action_logits.view((n_action_logits.shape[0], -1))
        n_dist = torch.distributions.OneHotCategorical(logits=flattened_n_action_logits)

        n_action_flat = n_dist.sample()
        n_log_prob = n_dist.log_prob(n_action_flat)

        n_action = n_action_flat.view(self.act_shape)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return n_action.cpu().detach().numpy(), n_log_prob.cpu().detach()
