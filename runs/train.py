import argparse
import os
import pickle
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from math import exp

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.agents import SharedPolicy, VDN, sharedpolicy, MohitoActor
from time import time 

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, s, ed, r, s_, ed_):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s, ed, r, s_, ed_)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, hyedge, reward, next_state, next_hyedge = map(np.stack, zip(*batch))
        return state, hyedge, reward, next_state, next_hyedge

from free_range_zoo.envs.wildfire.agents.mohito import build_incidence_graph 

def get_buffer_data(agent_obs, agent_hyperedge, rewards, agent_obs_next, agent_hyperedge_next):
    """
    Prepare the data for the replay buffer.
    Args:
        agent_obs: dict - Observations of the agents.
        agent_hyperedge: dict - Hyperedges of the agents.
        rewards: dict - Rewards received by the agents.
        agent_obs_next: dict - Next observations of the agents.
        agent_hyperedge_next: dict - Next hyperedges of the agents.
    Returns:
        tuple - Tuple containing state, hyperedge, reward, next_state, next_hyperedge.
    """
    inc_graph = build_incidence_graph(agent_obs['firefighter_1'][0], torch.cat([agent_obs['firefighter_2'],agent_obs['firefighter_3']]), agent_obs['tasks'][0])
    hyperedges = torch.stack([hy for hy in agent_hyperedge.values()])

    rewards_sum = torch.sum(torch.stack([r for r in rewards.values()]))

    inc_graph_next = build_incidence_graph(agent_obs_next['firefighter_1'][0], torch.cat([agent_obs_next['firefighter_2'],agent_obs_next['firefighter_3']]), agent_obs_next['tasks'][0])
    hyperedges_next = torch.stack([hy for hy in agent_hyperedge_next.values()])

    return inc_graph, hyperedges, rewards_sum, inc_graph_next, hyperedges_next

def train(args) -> None:
    """
    Train a shared policy for the wildfire environment using a simple Q-learning approach.
    Args:
        args: argparse.Namespace - Command line arguments.
    """
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load wildfire configuration from a pkl file.
    paths = [f"configs/wildfire/WS{i}.pkl" for i in [1,2,3]]
    config_WS = [pickle.load(open(config_path, "rb")) for config_path in paths]
    
    epsilon_fn = lambda episode: args.epsilon_init+ (args.epsilon_init - args.epsilon_min) * exp(-args.epsilon_decay * episode)

    envs = {}
    for idx in [1,2,3]:
        envs[idx] = wildfire_v0.parallel_env(
            parallel_envs=1,
            max_steps=args.max_steps,
            device=device,
            configuration=config_WS[idx-1],
            buffer_size=50,
            single_seeding=True,
            show_bad_actions=True,
            override_initialization_check=True,
        )
    step = 0

    loss_history = []
    reward_history = []

    for episode in range(args.num_episodes):
        step += 1
        # Create the parallel environment with a single environment for training.
        i = 1
        env = action_mapping_wrapper_v0(envs[i])

        agents = {
            env.agents[0]: MohitoActor(agent_name="firefighter_1", parallel_envs=1),
            env.agents[1]: MohitoActor(agent_name="firefighter_2", parallel_envs=1),
            env.agents[2]: MohitoActor(agent_name="firefighter_3", parallel_envs=1),
        }

        # Optimizers for actor and critic networks
        optimizers_actor = {}
        optimizers_critic = {}
        for agent_name, agent in agents.items():
            optimizers_actor[agent_name] = optim.Adam(agent.actor_graph.parameters(), lr=args.lr_actor)
            optimizers_critic[agent_name] = optim.Adam(agent.critic_graph.parameters(), lr=args.lr_critic)
        

        observations, _ = env.reset()
        total_reward = 0

        while not torch.all(env.finished):
            
            agent_obs = {}
            agent_obs_next = {}
            agent_hyperedge = {}
            agent_hyperedge_next = {}
            #  Get observations and hyperedge for each agent
            for agent_name, agent in agents.items():
                agent_obs[agent_name] = observations[agent_name][0]['self']
                agent_hyperedge[agent_name] = agent.observe(observations[agent_name], epsilon=epsilon_fn(episode))
            agent_obs['tasks'] = observations[agent_name][0]['tasks']

            #  Get actions for each agent
            agent_actions = {
                agent_name: agents[agent_name].act(
                    action_space=env.action_space(agent_name)
                )
                for agent_name in env.agents
            } 

            #  Step the environment with the actions
            observations, rewards, terminations, truncations, infos = env.step(
                agent_actions
            )

            total_reward += sum(rewards.values())

            #  Get next observations and hyperedge for each agent1
            for agent_name, agent in agents.items():
                agent_obs_next[agent_name] = observations[agent_name][0]['self']
                agent_hyperedge_next[agent_name] = agent.observe(observations[agent_name]) 
            agent_obs_next['tasks'] = observations[agent_name][0]['tasks']

            # Add the data to the buffer
            obs, hyperedges, reward, obs_next, hyperedges_next = get_buffer_data(
                agent_obs, agent_hyperedge, rewards, agent_obs_next, agent_hyperedge_next
            )

            for agent_name, agent in agents.items():
                q_value = agent.evalutate_critic(obs, hyperedges)
                q_value_next =  agent.evalutate_critic(obs_next, hyperedges_next)
                
                loss_critic = F.mse_loss(q_value, reward + args.gamma * q_value_next)
                optimizers_critic[agent_name].zero_grad()
                loss_critic.backward()
                optimizers_critic[agent_name].step()

                # Actor update
                q_value_actor = agent.evalutate_critic(obs, hyperedges)
                loss_actor = - q_value_actor
                optimizers_actor[agent_name].zero_grad()
                loss_actor.backward()
                optimizers_actor[agent_name].step()

        loss_history.append(loss_critic.item())
        reward_history.append(total_reward)
        #  Calculate total reward for the episode
        print(f"Episode {episode}: Total reward = {total_reward}, Steps = {step}, Loss = {loss_critic.item():.4f}, Epsilon = {epsilon_fn(episode):.4f}")


    # Loss visualization
    plt.figure()
    plt.plot(range(1, args.num_episodes+1), loss_history, marker='o')
    plt.title('Training Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    t = f"{time()}"
    plt.savefig(os.path.join(args.output,t,'loss_plot.png'))
    plt.show()

    # 학습된 모델 저장
    model_save_path = os.path.join(args.output, t,"vdn.pth")
    torch.save(agents[next(iter(agents))].policy.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Train Shared Policy for Wildfire Environment")
    parser.add_argument('--output', type=str, default='./output', help="Directory to save the trained model")
    # parser.add_argument('--config', type=str, default='WS1', help="Wildfire configuration (e.g., WS3)")
    parser.add_argument('--cuda', action='store_true', help="Use cuda if available")
    parser.add_argument('--num_episodes', type=int, default=10000, help="Number of training episodes")
    parser.add_argument('--max_steps', type=int, default=50, help="Maximum steps per episode")
    parser.add_argument('--lr_actor', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--lr_critic', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.9, help="Discount factor")
    # Exploration parameters for epsilon-greedy
    parser.add_argument('--epsilon_init', type=float, default=0.9, help="Initial epsilon for exploration")
    parser.add_argument('--epsilon_decay', type=float, default=0.00035, help="Decay rate for epsilon per episode")
    parser.add_argument('--epsilon_min', type=float, default=0.05, help="Minimum epsilon")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args)