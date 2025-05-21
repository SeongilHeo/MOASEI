import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

from core import Incidence_graph, GNNActor

base_path = "."
config_path = "competition_configs/wildfire"
wildfire_configs = ["WS1", "WS2", "WS3"]
envs = []

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

def save_model(epoch, network , option=None):
    torch.save(network, f'{base_path}/trained_model/model{option}.h5')

def load_model(model_path):
    return torch.load(model_path, map_location='cpu', weights_only=False)

def get_fire_outcomes_by_fuel(task_obs_t0, task_obs_t1, intensity_threshold=0.1):
    lookup = {
        (int(x1), int(y1)): fuel1
        for x1, y1, intensity1, fuel1 in task_obs_t1
    }

    fire_outcomes = []
    for x0, y0, intensity0, fuel0 in task_obs_t0:
        if intensity0 <= intensity_threshold:
            continue
        key = (int(x0), int(y0))
        fuel1 = lookup.get(key)
        fire_outcomes.append(1 if (fuel1 is None or fuel0 > fuel1) else 0)

    return fire_outcomes

def reward_to_go(rewards):
    arr = np.array(rewards)
    return np.cumsum(arr[::-1])[::-1].tolist()

def init_batch(agent_names):
    return {
        name: {
            'obs': [], 'acts': [], 'weights': [],
            'input': [], 'hmask': [], 'omask': [],
            'ohmask': [], 'other_obs': [], 'other_acts': [], 'outcome': [],
            'adv': [], 'loss': [], 'valid_logits': []
        }
        for name in agent_names
    }

load_config()

obs_dim = 4
hidden_dim = 32

num_envs = len(envs)
batch_size = 200
num_episodes = 50
# init_weights = load_model(f"{base_path}/trained_model/model_env3_200-30.h5")
# shared_actor = GNNActor(input_dim=obs_dim, hidden_dim=hidden_dim, init_weights = init_weights)
shared_actor = GNNActor(input_dim=obs_dim, hidden_dim=hidden_dim)
actor_optimizer = torch.optim.Adam(shared_actor.parameters(), lr=1e-2)

Graph  = Incidence_graph(node_size=obs_dim)

stores={'reward':[], 'loss':[]}

for epoch in range(num_episodes):
    env = envs[epoch % num_envs]

    for iii in range(5):
        observations, _ = env.reset()
        agent_names = env.agents

        batchs = init_batch(agent_names)

        rewards = []

        while True:
            task_t0 = {}
            env_actions = {}
            intents = {}

            for agent_name in agent_names:
                obs_dict, t_map = observations[agent_name]
                t_mapping = t_map['agent_action_mapping'][0]

                self_obs = obs_dict['self'][0]
                other_obs = obs_dict['others'][0]
                task_obs = obs_dict['tasks'].to_padded_tensor(-100)[0]

                data, hmask, omask, ohmask = Graph.build(self_obs, other_obs, task_obs, t_mapping)

                env_action, agent_actions, _, intent = shared_actor.forward_pass(data, hmask, t_mapping, omask)

                env_actions[agent_name] = env_action
                intents[agent_name] = intent

                batchs[agent_name]['input'].append(data)
                batchs[agent_name]['hmask'].append(hmask)
                batchs[agent_name]['omask'].append(omask)
                batchs[agent_name]['ohmask'].append(ohmask)
                batchs[agent_name]['obs'].append(t_mapping)
                batchs[agent_name]['acts'].append(agent_actions)
                batchs[agent_name]['other_obs'].append([
                    observations[other][0]['self'][0][3] # suppressant
                    for other in agent_names if other != agent_name
                ])

                task_t0[agent_name] = task_obs.tolist()


            
            observations, reward, terminations, truncations, infos = env.step(env_actions)

            rewards.append(reward['firefighter_1'])
            
            done = torch.all(env.finished)

            task_t1 = {}
            for name in agent_names:
                if done:
                    task_t1[name] = task_t0[name]
                else:
                    t_obs = observations[name][0]['tasks'].to_padded_tensor(-100)[0] # [num_tasks, obs_dim]
                    task_t1[name] = t_obs.tolist()

            for agent_name in agent_names:
                # Add other agents' actions to the batch
                batchs[agent_name]['other_acts'].append([
                    intents[other] 
                    for other in agent_names if other != agent_name
                ])
                # Compute fire outcomes (1 = affected, 0 = not affected)
                batchs[agent_name]['outcome'].append(
                    get_fire_outcomes_by_fuel(task_t0[agent_name], task_t1[agent_name])
                )

            if done:
                for agent_name in agent_names:
                    batchs[agent_name]['weights'] += list(reward_to_go(rewards))

                store = rewards
                done, rewards = False, []
                observations, infos = env.reset()

                if len(batchs[agent_name]['obs']) >= batch_size:
                    break

        # Zero, backward, and step only once per batch (shared weights!)
        actor_optimizer.zero_grad()
        

        losses = [
            shared_actor.compute_loss(
                batchs[agent_name]['input'],
                batchs[agent_name]['hmask'],
                batchs[agent_name]['acts'],
                batchs[agent_name]['weights'],
                batchs[agent_name]['omask'],
                batchs[agent_name]['other_obs'],
                batchs[agent_name]['ohmask'],
                batchs[agent_name]['other_acts'],
                batchs[agent_name]['outcome'],
            )
            for agent_name in agent_names
        ]
        total_loss = torch.stack(losses).sum()
        
        total_loss.backward()
        actor_optimizer.step()


        env.close()

        print(f"Epoch: {epoch}, Reward: {sum(store).item()}")
        stores['reward'].append(sum(store).item())
        # stores['actor_loss'].append(total_loss.item())
        
stores_np = np.array([s.item() if isinstance(s, torch.Tensor) else s for s in stores])
epochs = list(range(len(stores_np)))

save_model(epoch, shared_actor, "")
def plot_reward_curve(stores):
    plt.figure(figsize=(10, 5))
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
# plot_loss_curve(stores)