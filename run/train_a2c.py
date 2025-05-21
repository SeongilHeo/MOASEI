import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

from core import COMACritic, GNNActor, Incidence_graph

def load_config():
    global envs
    for wildfire_config in wildfire_configs:
        with open(f"{base_path}/{config_path}/{wildfire_config}.pkl", 'rb') as f:
            wildfire_configuration = pickle.load(f)

            env = wildfire_v0.parallel_env(
                max_steps=100,
                parallel_envs=1,
                configuration=wildfire_configuration,
                device=torch.device('cpu'),
                log_directory="test_logging",
                override_initialization_check=True
            )

            env = action_mapping_wrapper_v0(env)

        envs.append(env)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
def save_model(epoch, network , option=None):
    torch.save(network, f'{base_path}/trained_model/model{option}.h5')
def build_joint_action_tensor(action_history, action_dim):
    """
    Builds a one-step joint action tensor from a dict of agent actions.

    Args:
        action_history (dict): {agent_name: int action}, for a single timestep.
        action_dim (int): Number of possible discrete actions per agent.

    Returns:
        torch.Tensor: [1, num_agents * action_dim] one-hot vector.
    """
    actions = torch.tensor(action_history)
    one_hots = torch.eye(action_dim)[actions]  # Shape: [num_agents, action_dim]
    return one_hots.flatten().unsqueeze(0)  # Shape: [1, num_agents * action_dim]

base_path = "."
config_path = "competition_configs/wildfire"
wildfire_configs = ["WS1", "WS2", "WS3"]
envs = []

def init_batch(agent_names):
    return {
        name: {
            'obs': [], 'acts': [], 'weights': [],
            'input': [], 'hmask': [],# 'omask': [], 
            'ohmask': [], 'other_obs': [], 'other_acts': [], 'outcome': [],
            'adv': [], 'loss': [], 'logits': []
        }
        for name in agent_names
    }

load_config()

obs_dim = 4
hidden_dim = 64

num_envs = len(envs)
num_agents = 3
action_dim = 6 

num_episodes = 150

shared_actor = GNNActor(input_dim=obs_dim, hidden_dim=hidden_dim)
actor_optimizer = torch.optim.Adam(shared_actor.parameters(), lr=1e-2)

central_critic = COMACritic(input_dim=4, hidden_dim=hidden_dim, action_dim=action_dim+1, num_agents=num_agents)
critic_optimizer = torch.optim.Adam(central_critic.parameters(), lr=1e-2)

Graph  = Incidence_graph(node_size=obs_dim, all_node=False)

stores={'reward':[], 'actor_loss':[], 'critic_loss':[]}

batch_size=150

for epoch in range(num_episodes):
    env = envs[epoch // 50]
    agent_names = env.agents

    batchs = init_batch(agent_names)
    batch_joint_rewards = [] 
    batch_joint_data = []
    batch_joint_actions = []
        
    rewards= []

    observations, _ = env.reset()

    while True:
        env_actions = {}

        all_self_obs = []
        all_t_mapping = []
        all_agent_actions = []

        for agent_name in agent_names:
            obs_dict, t_map = observations[agent_name]
            t_mapping = t_map['agent_action_mapping'][0]

            self_obs = obs_dict['self'][0]
            other_obs = obs_dict['others'][0]
            task_obs = obs_dict['tasks'].to_padded_tensor(-100)[0]

            data, hmask, _, _ = Graph.build(self_obs, other_obs, task_obs, t_mapping)

            env_action, agent_actions, logits, _ = shared_actor.forward_pass(data, hmask, t_mapping)

            env_actions[agent_name] = env_action

            batchs[agent_name]['input'].append(data)
            batchs[agent_name]['hmask'].append(hmask)
            batchs[agent_name]['obs'].append(t_mapping)
            batchs[agent_name]['acts'].append(agent_actions)
            batchs[agent_name]['logits'].append(logits)

            all_self_obs.append(self_obs)
            all_t_mapping.append(t_mapping)
            all_agent_actions.append(agent_actions)

        joint_data = Graph.build_joint(all_self_obs, task_obs, all_t_mapping)
        batch_joint_data.append(joint_data)
        

        joint_action=build_joint_action_tensor(all_agent_actions, action_dim+1)
        batch_joint_actions.append(joint_action)

        for index, agent_name in enumerate(agent_names):
            batchs[agent_name]['adv'].append(
                central_critic.compute_advantage(
                    index, joint_data, joint_action, batchs[agent_name]['logits'][-1])
            )

        observations, reward, terminations, truncations, infos = env.step(env_actions)

        done = torch.all(env.finished)

        rewards.append(sum(reward.values())/num_agents)
        
        if done:
            joint_rewards = list(discount_cumsum(rewards, 1))
            batch_joint_rewards += joint_rewards

            for agent_name in agent_names:
                batchs[agent_name]['weights'] += list(batchs[agent_name]['adv'])

            store = rewards
            done, rewards, all_t_mapping = False, [], []
            observations, infos = env.reset()

            for agent_name in agent_names:
                batchs[agent_name]['adv'] = []
            if len(batchs[agent_name]['obs']) >= batch_size:
                break
        
    actor_optimizer.zero_grad()

    losses = [
        shared_actor.compute_loss(
            batchs[agent_name]['input'],
            batchs[agent_name]['hmask'],
            batchs[agent_name]['acts'],
            batchs[agent_name]['weights'],
            λ_suppress=0,
            λ_intent=0,
            λ_belief=0
        )
        for agent_name in agent_names
    ]
    total_loss = torch.stack(losses).sum()
    
    total_loss.backward()
    actor_optimizer.step()

    env.close()

    for n in range(8):
        critic_optimizer.zero_grad()
        critic_loss=central_critic.compute_loss(batch_joint_data, batch_joint_actions, batch_joint_rewards)
        critic_loss.backward()
        critic_optimizer.step()

    print(f"Epoch: {epoch}, Reward: {sum(store).item()}")

    stores['reward'].append(sum(store).item())
    stores['actor_loss'].append(total_loss.item())
    stores['critic_loss'].append(critic_loss.item())

def plot_reward_curve(stores):
    # convert to numpy array and define epochs
    stores_np = np.array([s.item() if isinstance(s, torch.Tensor) else s for s in stores])
    epochs = np.arange(len(stores_np))
    # plot raw rewards
    plt.plot(epochs, stores_np, color='C0', alpha=0.3, label='Reward')
    # compute moving average
    window_size = 5
    mov_avg = np.convolve(stores_np, np.ones(window_size)/window_size, mode='same')
    plt.plot(epochs, mov_avg, color='C1', label=f'Moving Avg')
    # compute rolling min and max over a window and shade the range
    window_size = 5
    roll_min = np.array([stores_np[max(0, i-window_size+1):i+1].min() for i in range(len(stores_np))])
    roll_max = np.array([stores_np[max(0, i-window_size+1):i+1].max() for i in range(len(stores_np))])
    # shade reward range (min-max)
    plt.fill_between(epochs, roll_min, roll_max, color='C0', alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reward_curve.png')
    plt.show()

def plot_loss_curve(stores):
    # prepare epochs
    epochs = np.arange(len(stores['actor_loss']))
    # create figure and primary axis for critic loss
    fig, ax1 = plt.subplots()
    color_c = 'red'
    ax1.plot(epochs, stores['critic_loss'], color=color_c, label='Critic Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Critic Loss', color=color_c)
    ax1.tick_params(axis='y', labelcolor=color_c)
    ax1.grid(True)
    # create secondary axis for actor loss
    ax2 = ax1.twinx()
    color_a = 'blue'
    ax2.plot(epochs, stores['actor_loss'], color=color_a, label='Actor Loss')
    ax2.set_ylabel('Actor Loss', color=color_a)
    ax2.tick_params(axis='y', labelcolor=color_a)
    # combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    plt.title('Loss Curve')
    fig.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()

plot_reward_curve(stores['reward'])
plot_loss_curve(stores)

save_model(epoch, shared_actor, "_a2c")
