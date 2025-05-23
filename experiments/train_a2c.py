import torch
import argparse
import logging
import os
from datetime import datetime

from core import Incidence_graph, GNNActor, COMACritic
from experiments.utils import (
    load_configs,
    init_batch,
    discount_cumsum,
    build_joint_action_tensor,
    load_model,
    save_model,
    plot_reward_curve,
    plot_loss_curve,
    FORMAT_STRING
)

def train():
    """
    Train the GNN-based A2C (Advantage Actor Critic) model on a set of environments.
    """


    args = handle_args()  # parse command-line arguments

    # Unpack hyperparameters from args
    base_path=args.base_path
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_episodes = args.num_episodes
    batch_size = args.batch_size
    curriculum = args.curriculum
    init_model = args.init_model

    # Set up logging
    if base_path is not None:
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    ENVS = load_configs(logging_path=base_path) # load environment configurations from file or directory
    num_envs = len(ENVS)
    num_agents = 3
    action_dim = 6  # number of discrete actions

    train_logger.setLevel(args.log_level)

    # Instantiate shared GNN-based actor and its optimizer
    if init_model:
        init_weights = load_model(init_model)
        shared_actor = GNNActor(input_dim=input_dim, hidden_dim=hidden_dim, init_weights=init_weights)
    else:
        shared_actor = GNNActor(input_dim=input_dim, hidden_dim=hidden_dim)

    actor_optimizer = torch.optim.Adam(shared_actor.parameters(), lr=1e-2)

    # Instantiate centralized critic and its optimizer
    central_critic = COMACritic(
        input_dim=4,
        hidden_dim=hidden_dim,
        action_dim=action_dim + 1,
        num_agents=num_agents
    )
    critic_optimizer = torch.optim.Adam(central_critic.parameters(), lr=1e-2)

    # Build an incidence graph to process observations
    Graph = Incidence_graph(node_size=input_dim, all_node=False)

    # Storage for plotting curves later
    stores = {'reward': [], 'actor_loss': [], 'critic_loss': []}

    for epoch in range(num_episodes):
        # Select environment according to curriculum or round-robin
        if curriculum:
            env = ENVS[epoch // (num_episodes // num_envs)]
        else:
            env = ENVS[epoch % num_envs]

        agent_names = env.agents

        # Initialize per-agent minibatch buffers
        batchs = init_batch(agent_names)
        batch_joint_data = []
        batch_joint_actions = []
        batch_joint_rewards = []

        rewards = []
        observations, _ = env.reset()  # reset environment and get initial obs

        # Collect trajectories until batch_size is reached
        while True:
            env_actions = {}
            all_self_obs = []
            all_t_mapping = []
            all_agent_actions = []

            # Loop through each agent to select actions
            for agent_name in agent_names:
                obs_dict, t_map = observations[agent_name]
                t_mapping = t_map['agent_action_mapping'][0]

                # Extract self, others, and task observations
                self_obs = obs_dict['self'][0]
                other_obs = obs_dict['others'][0]
                task_obs = obs_dict['tasks'].to_padded_tensor(-100)[0]

                # Build graph inputs for GNNActor
                data, hmask, _, _ = Graph.build(self_obs, other_obs, task_obs, t_mapping)

                # Forward pass through shared actor
                env_action, agent_actions, logits, _ = shared_actor.forward_pass(
                    data, hmask, t_mapping
                )
                env_actions[agent_name] = env_action

                # Store per-agent data
                batchs[agent_name]['input'].append(data)
                batchs[agent_name]['hmask'].append(hmask)
                batchs[agent_name]['obs'].append(t_mapping)
                batchs[agent_name]['acts'].append(agent_actions)
                batchs[agent_name]['logits'].append(logits)

                all_self_obs.append(self_obs)
                all_t_mapping.append(t_mapping)
                all_agent_actions.append(agent_actions)

            # Build joint representation across agents for the critic
            joint_data = Graph.build_joint(all_self_obs, task_obs, all_t_mapping)
            batch_joint_data.append(joint_data)

            # Combine discrete actions into a joint tensor
            joint_action = build_joint_action_tensor(all_agent_actions, action_dim + 1)
            batch_joint_actions.append(joint_action)

            # Compute and store advantage for each agent
            for idx, agent_name in enumerate(agent_names):
                adv = central_critic.compute_advantage(
                    idx, joint_data, joint_action, batchs[agent_name]['logits'][-1]
                )
                batchs[agent_name]['adv'].append(adv)

            # Step the environment with the chosen actions
            observations, reward, _, _, _ = env.step(env_actions)
            done = torch.all(env.finished)  # check if all agents are done
            rewards.append(sum(reward.values()) / num_agents)

            if done:
                # Compute discounted cumulative rewards
                joint_rewards = list(discount_cumsum(rewards, 1))
                batch_joint_rewards += joint_rewards

                # Set the weights for each agent's policy loss
                for agent_name in agent_names:
                    batchs[agent_name]['weights'] += list(batchs[agent_name]['adv'])

                store = rewards  # store for logging
                # Reset buffers and environment
                done, rewards = False, []
                observations, _ = env.reset()
                for agent_name in agent_names:
                    batchs[agent_name]['adv'] = []

                # Break once enough data has been collected
                if len(batchs[agent_name]['obs']) >= batch_size:
                    break

        # Update actor: compute policy loss and backpropagate
        actor_optimizer.zero_grad()
        losses = [
            shared_actor.compute_loss(
                batchs[name]['input'],
                batchs[name]['hmask'],
                batchs[name]['acts'],
                batchs[name]['weights'],
                λ_suppress=0,
                λ_intent=0,
                λ_belief=0
            )
            for name in agent_names
        ]
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        actor_optimizer.step()

        env.close()

        # Update critic multiple times per epoch
        for _ in range(8):
            critic_optimizer.zero_grad()
            critic_loss = central_critic.compute_loss(
                batch_joint_data,
                batch_joint_actions,
                batch_joint_rewards
            )
            critic_loss.backward()
            critic_optimizer.step()

        # Logging
        epoch_reward = sum(store).item()
        train_logger.info(f"Epoch: {epoch}, Reward: {epoch_reward}")
        stores['reward'].append(epoch_reward)
        stores['actor_loss'].append(total_loss.item())
        stores['critic_loss'].append(critic_loss.item())

    # Plot training curves and save model
    plot_reward_curve(stores['reward'], base_path=base_path)
    plot_loss_curve(stores['actor_loss'], stores['critic_loss'], base_path=base_path)
    save_model(shared_actor, base_path=base_path, postfix="_a2c")


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
        help='path to initial model weights'
    )
    parser.add_argument(
        '--log_level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
        default='INFO', 
        help='Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default=f"logging/{datetime.now().strftime("%y%m%d_%H%M%S")}",
        help='Base path for saving models and logs (defaults to current time yymmdd:hhmmss)'
    )
    return parser.parse_args()

logging.basicConfig(level=logging.INFO, format=FORMAT_STRING)
train_logger = logging.getLogger('train')

if __name__ == "__main__":
    train()