import os
import pickle
import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt

from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0

def load_configs(base_path=".", config_path = "competition_configs/wildfire"):
    wildfire_configs = ["WS1", "WS2", "WS3"]

    envs = []

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

    return envs

def init_batch(agent_names):
    return {
        name: {
            'obs': [], 'acts': [], 'weights': [],
            'input': [], 'hmask': [], 'omask': [], 
            'ohmask': [], 'other_obs': [], 'other_acts': [], 'outcome': [],
            'adv': [], 'loss': [], 'logits': []
        }
        for name in agent_names
    }

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

def save_model(network , base_path=".", postfix=None):
    models_dir = os.path.join(base_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    filename = f"model{postfix}.h5"
    filepath = os.path.join(models_dir, filename)

    while os.path.exists(filepath):
        new_name = input(f"'{filename}' already exists. Enter a new model name (without extension): ")
        filename = f"{new_name}.h5"
        filepath = os.path.join(models_dir, filename)

    torch.save(network, filepath)

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

def plot_reward_curve(stores, save=True):
    """
    Plots the reward curve over epochs, including:
      - Raw reward per epoch (light line)
      - Moving average (solid line)
      - Shaded area between rolling minimum and maximum

    Args:
        stores (list or np.ndarray or torch.Tensor):
            Sequence of reward values for each epoch.
    """
    # Convert rewards to a numpy array of floats
    # If an element is a torch.Tensor, extract its scalar via .item()
    stores_np = np.array([
        s.item() if isinstance(s, torch.Tensor) else s
        for s in stores
    ])
    epochs = np.arange(len(stores_np))

    # Plot raw reward values (semi-transparent)
    plt.plot(
        epochs, stores_np,
        color='C0', alpha=0.3,
        label='Reward'
    )

    # Compute and plot the moving average (window_size=5)
    window_size = 5
    mov_avg = np.convolve(
        stores_np,
        np.ones(window_size) / window_size,
        mode='same'
    )
    plt.plot(
        epochs, mov_avg,
        color='C1',
        label=f'Moving Avg (w={window_size})'
    )

    # Compute rolling minimum and maximum over the same window
    roll_min = np.array([
        stores_np[max(0, i - window_size + 1):i + 1].min()
        for i in range(len(stores_np))
    ])
    roll_max = np.array([
        stores_np[max(0, i - window_size + 1):i + 1].max()
        for i in range(len(stores_np))
    ])

    # Shade the area between rolling min and max
    plt.fill_between(
        epochs,
        roll_min,
        roll_max,
        color='C0',
        alpha=0.2,
        label='Min–Max Range'
    )

    # Final plot adjustments
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file and display
    if save:
        plt.savefig('reward_curve.png')
    plt.show()

def plot_loss_curve(actor_loss, critic_loss, save=True):
    """
    Plots actor and critic loss curves on shared x-axis with twin y-axes.

    Args:
        actor_loss (list or np.ndarray): Sequence of actor losses per epoch.
        critic_loss (list or np.ndarray): Sequence of critic losses per epoch.
    """
    # Ensure arrays for indexing
    actor_loss = np.array(actor_loss)
    critic_loss = np.array(critic_loss)

    # Epochs based on length of losses
    epochs = np.arange(len(actor_loss))

    # Create figure and primary axis for critic loss
    fig, ax1 = plt.subplots()
    color_c = 'red'
    ax1.plot(epochs, critic_loss, color=color_c, label='Critic Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Critic Loss', color=color_c)
    ax1.tick_params(axis='y', labelcolor=color_c)
    ax1.grid(True)

    # Secondary axis for actor loss
    ax2 = ax1.twinx()
    color_a = 'blue'
    ax2.plot(epochs, actor_loss, color=color_a, label='Actor Loss')
    ax2.set_ylabel('Actor Loss', color=color_a)
    ax2.tick_params(axis='y', labelcolor=color_a)

    # Combine legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')

    plt.title('Loss Curve')
    fig.tight_layout()
    if save:
        plt.savefig('loss_curve.png')
    plt.show()
