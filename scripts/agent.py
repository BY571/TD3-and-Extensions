import torch
import numpy as np
import random
from .networks import Actor, Critic
import torch.optim as optim
import torch.nn.functional as F

class TD3_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 state_size,
                 action_size,
                 action_low,
                 action_high,
                 replay_buffer,
                 batch_size,
                 random_seed,
                 lr,
                 hidden_size,
                 gamma,
                 tau,
                 device):
        """Initialize an Agent object

        Args:
            state_size (int):    State size
            action_size (int):   Action size
            replay_buffer:       Experience Replay Buffer
            batch_size:          Batch size when learning
            random_seed (int):   Random seed
            lr (float):          Learning rate
            hidden_size (int):   Number of hidden units per layer
            gamma (float):       Discount factor
            tau (float):         Tau, soft-update parameter
            device (torch device): Training Device cpu or cuda:0
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.seed = random.seed(random_seed)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
                
        print("Using: ", device)
        self.device = device
        self.iter = 1
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed+1, hidden_size=hidden_size).to(device)
        self.target_critic1 = Critic(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.target_critic2 = Critic(state_size, action_size, random_seed+1, hidden_size=hidden_size).to(device)

        self.optimizer1 = optim.Adam(self.critic1.parameters(), lr=lr, weight_decay=0)
        self.optimizer2 = optim.Adam(self.critic2.parameters(), lr=lr, weight_decay=0)

        # Replay memory
        self.memory = replay_buffer
        

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)


        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        mu = self.actor_local(state)
        action = mu + torch.normal(mean=torch.FloatTensor([0.]),
                                   std=torch.FloatTensor([0.1])).to(mu.device)
        return action.detach().cpu()[0]
    
    def eval(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        mu = self.actor_local(state)
        return mu.detach().cpu()[0]
    
    def learn(self,experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

       

        # ---------------------------- update critic ---------------------------- #

        with torch.no_grad():
            # Get predicted next-state actions and Q values from target models
            next_action = self.actor_local(next_states)
            next_action = next_action + torch.clamp(
                                                torch.normal(mean=torch.FloatTensor([0.]),
                                                             std=torch.FloatTensor([0.2])),
                                                min=-0.5, max=0.5).to(next_action.device)
            next_action = torch.clamp(next_action, min=self.action_low, max=self.action_high)
            # TODO: make this variable for possible more than two target critics
            Q_target1_next = self.target_critic1(next_states, next_action.squeeze(0))
            Q_target2_next = self.target_critic2(next_states, next_action.squeeze(0))
            
            # take the min of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        Q_targets = rewards.cpu() + (self.gamma * (1 - dones.cpu()) * Q_target_next.cpu())

        # Compute critic losses and update critics 

        Q1 = self.critic1(states, actions).cpu()
        Q2 = self.critic2(states, actions).cpu()
        Q1_loss = 0.5*F.mse_loss(Q1, Q_targets)
        Q2_loss = 0.5*F.mse_loss(Q2, Q_targets)
    
        # Update critic
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        Q1_loss.backward()
        Q2_loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()


        # ---------------------------- update actor ---------------------------- #
        if self.iter % 2 == 0:

            actions_pred = self.actor_local(states)             
            
            Q1 = self.critic1(states, actions_pred.squeeze(0)).cpu()
            #Q2 = self.critic2(states, actions_pred.squeeze(0)).cpu()
            #Q = torch.min(Q1,Q2)

            actor_loss =  -(Q1).mean() # * weights
            # Optimize the actor loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # soft update of the targets
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)
            self.iter = 0

        self.iter +=1
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
