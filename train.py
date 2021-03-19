from scripts.buffer import ReplayBuffer, PrioritizedReplay
import gym
import random
import numpy as np 
import torch 
import pybullet_envs # to run e.g. HalfCheetahBullet-v0 different reward function bullet-v0 starts ~ -1500. pybullet-v0 starts at 0
from collections import deque
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from scripts.agent import TD3_Agent

def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def evaluate(step, eval_runs=5, capture=False):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()

        rewards = 0
        while True:
            action = agent.eval(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)
            state, reward, done, _ = eval_env.step(action_v)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if capture == False:   
        writer.add_scalar("Test_Reward", np.mean(reward_batch), step)

def fill_buffer(agent, env, samples=1000):
    collected_samples = 0
    
    state = env.reset()
    state = state.reshape((1, state_size))
    for i in range(samples):
            
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape((1, state_size))
        agent.memory.add(state, action, reward, next_state, done)
        collected_samples += 1
        state = next_state
        if done:
            state = env.reset()
            state = state.reshape((1, state_size))
    print("Added random samples to the Buffer - Buffer size: ", agent.memory.__len__())

    
def train(args):
    scores_deque = deque(maxlen=100)
    average_100_scores = []
    scores = []
    i_episode = 1
    state = env.reset()
    state = state.reshape((1, state_size))
    score = 0
    steps = args.steps
    for step in range(1, steps+1):
        # eval runs
        if step % args.eval_every == 0 or step == 1:
            evaluate(step, args.eval_runs)


        action = agent.act(state)
        action_v = action.numpy()
        action_v = np.clip(action_v, action_low, action_high)
        next_state, reward, done, info = env.step(action_v)
        next_state = next_state.reshape((1, state_size))
        agent.step(state, action, reward, next_state, done)
        
        state = next_state
        score += reward

        if done:
            scores_deque.append(score)
            scores.append(score)
            average_100_scores.append(np.mean(scores_deque))
            writer.add_scalar("Average100", np.mean(scores_deque), step)
            writer.add_scalar("Train_Reward", score, step)
            state = env.reset()
            state = state.reshape((1, state_size))
            
            print('\rEpisode {} Env. Step: [{}/{}] Train-Reward: {:.2f}  Average100 Score: {:.2f} '.format(i_episode, step, steps, score, np.mean(scores_deque)), end="")
            if i_episode % args.print_every == 0:
                print('\rEpisode {} Env. Step: [{}/{}] Train-Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, step, steps, score, np.mean(scores_deque)))
            score = 0 
            i_episode += 1

        
    return np.mean(scores_deque)


parser = argparse.ArgumentParser(description="")
parser.add_argument("--env", type=str,default="HalfCheetahBulletEnv-v0", help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("--info", type=str, default="TD3-training-run-1", help="Information or name of the run")
parser.add_argument("--steps", type=int, default=1_000_000, help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("--collect_random", type=int, default=5_000, help="Collect transitions of the envrionment with a random policy before training, default is 5000 transitions")
parser.add_argument("--eval_every", type=int, default=10_000, help="Number of interactions after which the evaluation runs are performed, default = 10.000")
parser.add_argument("--eval_runs", type=int, default=1, help="Number of evaluation runs performed, default = 1")
parser.add_argument("--seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("--nstep", type=int, default=1, help="Using Multistep Q-Learning, default n-step is 1")
parser.add_argument("--per", type=int, default=0, choices=[0,1], help="Using Prioritized Experience Replay if set to 1, default is 0")
parser.add_argument("--lr", type=float, default=3e-4, help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("--layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=0.005, help="Softupdate factor tau, default is 0.005")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--print_every", type=int, default=100, help="Print recent training results every x epochs, defaut is 100")

args = parser.parse_args()


if __name__ == "__main__":
    

    writer = SummaryWriter("runs/"+args.info)
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    action_high = env.action_space.high[0]
    seed = args.seed
    action_low = env.action_space.low[0]
    torch.manual_seed(seed)
    env.seed(seed)
    eval_env.seed(seed+1)
    np.random.seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.per == 0:
        replay_buffer = ReplayBuffer(buffer_size=args.replay_memory,
                                    batch_size=args.batch_size,
                                    seed=seed,
                                    gamma=args.gamma,
                                    n_step=args.nstep,
                                    device=device)
    else:
        replay_buffer = PrioritizedReplay(capacity=args.replay_memory,
                                          batch_size=args.batch_size,
                                          device=device,
                                          seed=seed,
                                          gamma=args.gamma,
                                          beta_frames=args.steps,
                                          n_step=args.nstep)
    agent = TD3_Agent(args=args,
                state_size=state_size,
                action_size=action_size,
                action_low=action_low,
                action_high=action_high,
                replay_buffer=replay_buffer,
                device=device
                )

    fill_buffer(agent, env=env, samples=args.collect_random)
    
    t0 = time.time()
    final_average100 = train(args)
    t1 = time.time()
    env.close()
    timer(t0, t1)
    
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    hparams = vars(args)
    metric = {"final average 100 train reward": final_average100}
    writer.add_hparams(hparams, metric)
