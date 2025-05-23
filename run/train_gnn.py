import torch
import argparse

from core import Incidence_graph, GNNActor
from run.utils import (
    load_configs,
    init_batch,
    reward_to_go,
    get_fire_outcomes_by_fuel,
    load_model,
    save_model,
    plot_reward_curve,
    plot_losses_curve
)

def train():
    """
    Train the GNN-based Policy Gradient model on a set of environments.
    """
    args = handle_args()  # parse command-line arguments
    ENVS = load_configs() # load environment configurations from file or directory

    num_envs = len(ENVS)
    num_agents = 3

    # Unpack hyperparameters from args
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    num_episodes = args.num_episodes
    curriculum = args.curriculum
    init_model = args.init_model

    # Instantiate shared GNN-based actor and its optimizer
    if init_model:
        init_weights = load_model(init_model)
        shared_actor = GNNActor(input_dim=input_dim, hidden_dim=hidden_dim, init_weights=init_weights)
    else:
        shared_actor = GNNActor(input_dim=input_dim, hidden_dim=hidden_dim)

    actor_optimizer = torch.optim.Adam(shared_actor.parameters(), lr=1e-2)

    # Build an incidence graph to process observations
    Graph = Incidence_graph(node_size=input_dim)

    # Storage for plotting curves later
    stores = {'reward': [], 'loss': []}

    for epoch in range(num_episodes):
        # Select environment according to curriculum or round-robin
        if curriculum:
            env = ENVS[epoch // (num_episodes // num_envs)]
        else:
            env = ENVS[epoch % num_envs]
        
        agent_names = env.agents

        for iii in range(5):

            # Initialize per-agent minibatch buffers
            batchs = init_batch(agent_names)

            rewards = []
            observations, _ = env.reset() # reset environment and get initial obs
            
            # Collect trajectories until batch_size is reached
            while True:
                task_t0 = {}
                env_actions = {}
                intents = {}
                
                # Loop through each agent to select actions
                for agent_name in agent_names:
                    obs_dict, t_map = observations[agent_name]
                    t_mapping = t_map['agent_action_mapping'][0]

                    # Extract self, others, and task observations
                    self_obs = obs_dict['self'][0]
                    other_obs = obs_dict['others'][0]
                    task_obs = obs_dict['tasks'].to_padded_tensor(-100)[0]
                    
                    # Build graph inputs for GNNActor
                    data, hmask, omask, ohmask = Graph.build(self_obs, other_obs, task_obs, t_mapping)
                    
                    # Forward pass through shared actor
                    env_action, agent_actions, _, intent = shared_actor.forward_pass(
                        data, hmask, t_mapping, omask
                    )
                    env_actions[agent_name] = env_action
                    
                    # Store per-agent data
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

                # Step the environment with the chosen actions
                observations, reward, _, _, _ = env.step(env_actions)
                done = torch.all(env.finished)
                rewards.append(sum(reward.values()) / num_agents)

                # Store the task observations for outcome calculation
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
                    # Compute discounted cumulative rewards
                    for agent_name in agent_names:
                        batchs[agent_name]['weights'] += list(reward_to_go(rewards))

                    store = rewards  # store for logging
                    # Reset buffers and environment
                    done, rewards = False, []
                    observations, _ = env.reset()

                    # Break once enough data has been collected
                    if len(batchs[agent_name]['obs']) >= batch_size:
                        break

            # Update actor: compute policy loss and backpropagate
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
            
            # Logging
            epoch_reward = sum(store).item()
            print(f"Epoch: {epoch}, Reward: {epoch_reward}")
            
    plot_reward_curve(stores['reward'])
    plot_losses_curve()
    save_model(shared_actor, "_pl")

def handle_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training script.
    """
    parser = argparse.ArgumentParser(
        description='Train A2C on wildfire configurations.'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=4,
        help='input dimension for the model'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='hidden dimension for the model'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=150,
        help='number of episodes to train'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch size for training'
    )
    parser.add_argument(
        '--curriculum',
        type=bool,
        default=False,
        help='curriculum learning flag (use easier envs first)'
    )
    parser.add_argument(
        '--init_model',
        type=str,
        default=None,
        help='path to the initial model weights'
    )
    return parser.parse_args()


if __name__ == "__main__":
    train()