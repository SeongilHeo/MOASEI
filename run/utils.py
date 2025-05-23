import os
import numpy as np
import torch

FORMAT_STRING = "[%(asctime)s] [%(levelname)8s] [%(name)10s] [%(filename)21s:%(lineno)03d] %(message)s"

def load_configs(logging_path='train_log',config_path="competition_configs/wildfire"):
    """
    Load and initialize multiple wildfire environments from pickled configuration files.

    Args:
        base_path (str): Root directory containing the configuration folder.
        config_path (str): Relative path under base_path where config .pkl files live.
    Returns:
        list: A list of parallel environments wrapped with action_mapping_wrapper_v0.
    """
    import pickle

    from free_range_zoo.envs import wildfire_v0
    from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
    wildfire_configs = ["WS1", "WS2", "WS3"]
    envs = []

    for wildfire_config in wildfire_configs:
        # Load the pickled configuration for each scenario
        cfg_file = os.path.join(config_path, f"{wildfire_config}.pkl")
        with open(cfg_file, 'rb') as f:
            wildfire_configuration = pickle.load(f)

        # Create a parallel wildfire environment with the loaded configuration
        env = wildfire_v0.parallel_env(
            max_steps=100,
            parallel_envs=1,
            configuration=wildfire_configuration,
            device=torch.device('cpu'),
            log_directory=f"{logging_path}/csv",
            override_initialization_check=True
        )
        # Apply the action-mapping wrapper to convert discrete actions
        env = action_mapping_wrapper_v0(env)

        envs.append(env)

    return envs


def init_batch(agent_names):
    """
    Initialize a batch data structure for each agent to store trajectories.

    Args:
        agent_names (iterable of str): Names or identifiers of agents.
    Returns:
        dict: Mapping from each agent name to a dict of empty lists for tracking:
            - obs: observations
            - acts: actions taken
            - weights: importance weights (rewards)
            - input: model inputs
            - hmask, omask, ohmask: various masks
            - other_obs, other_acts: data from other agents
            - outcome: final rewards or outcomes
            - adv: computed advantages
            - loss: per-step loss values
            - logits: raw policy network outputs
    """
    return {
        name: {
            'obs': [], 'acts': [], 'weights': [],
            'input': [], 'hmask': [], 'omask': [], 'ohmask': [],
            'other_obs': [], 'other_acts': [], 'outcome': [],
            'adv': [], 'loss': [], 'logits': []
        }
        for name in agent_names
    }

def discount_cumsum(x, discount):
    """
    Compute the discounted cumulative sum of a 1D sequence.

    Args:
        x (array_like): Input sequence of length N.
        discount (float): Discount factor (typically in [0, 1]).
    Returns:
        numpy.ndarray: Array of discounted cumulative sums with the same shape as x.
    Examples:
        >>> x = [1, 2, 3]
        >>> discount_cumsum(x, discount=0.5)
        array([1. + 0.5*2 + 0.5**2*3,
               2. + 0.5*3,
               3.])
    """
    from scipy.signal import lfilter

    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def get_fire_outcomes_by_fuel(task_obs_t0, task_obs_t1, intensity_threshold=0.1):
    """
    Determine whether fire at each location has decreased fuel or disappeared
    between two observations.

    Args:
        task_obs_t0 (iterable of tuple): Each tuple contains
            (x: float, y: float, intensity: float, fuel: numeric)
            observations at initial time t0.
        task_obs_t1 (iterable of tuple): Each tuple contains
            (x: float, y: float, intensity: float, fuel: numeric)
            observations at later time t1.
        intensity_threshold (float, optional): Minimum intensity at t0
            required to consider a location as burning. Defaults to 0.1.
    Returns:
        list of int: A list of outcome flags for each burning cell at t0:
            1 if the cell is missing in t1 or its fuel has decreased,
            otherwise 0.
    """
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
    """
    Compute reward-to-go for each timestep in a trajectory.

    Args:
        rewards (list or array-like): Sequence of scalar rewards [r₀, r₁, …, rₜ].
    Returns:
        list: reward-to-go for each timestep, i.e.
              [r₀ + r₁ + … + rₜ, r₁ + … + rₜ, …, rₜ].
    """
    # Convert to numpy array for efficient cumulative sum
    arr = np.array(rewards)
    # Reverse, compute cumulative sum, then reverse back to get reward-to-go
    return np.cumsum(arr[::-1])[::-1].tolist()

def load_model(model_path):
    """
    Load a PyTorch model checkpoint from disk.

    Args:
        model_path (str): Path to the saved model file.
    Returns:
        The loaded model object on CPU.
    """
    # torch.load will handle deserialization and map tensors to CPU
    return torch.load(model_path, map_location='cpu', weights_only=False)


def save_model(network, base_path=".", postfix=None):
    """
    Save a PyTorch model to a uniquely named file under `base_path/models`.

    Args:
        network (torch.nn.Module): The model to save.
        base_path (str): Root directory for saving (default: current working directory).
        postfix (str or None): Optional string to append to the filename.
    """
    # Ensure the models directory exists
    models_dir = os.path.join(base_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Build initial filename
    filename = f"model{postfix}.h5"
    filepath = os.path.join(models_dir, filename)

    # If the file already exists, prompt the user for a new name
    while os.path.exists(filepath):
        new_name = input(f"'{filename}' already exists. Enter a new model name (without extension): ")
        filename = f"{new_name}.h5"
        filepath = os.path.join(models_dir, filename)

    # Serialize the entire network object to disk
    torch.save(network, filepath)
    print(f"Model saved to {filepath}")

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

import matplotlib.pyplot as plt

def plot_reward_curve(stores, output_path='reward_curve.png', save=True, base_path='.'):
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
        path = os.path.join(base_path, output_path)
        plt.savefig(path)
        print(f"Saved reward curve to {path}")
    plt.show()

def plot_loss_curve(actor_loss, critic_loss, output_path='loss_curve.png', save=True, base_path='.'):
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
        path = os.path.join(base_path, output_path)
        plt.savefig(path)
        print(f"Saved loss curve to {path}")
    plt.show()

def plot_losses_curve(csv_path='losses.csv', output_path='losses_curve.png', save=True, base_path='.'):
    """
    Reads a CSV of logged losses and visualizes each loss component over training steps.

    Args:
        csv_path (str): Path to the CSV file containing loss data.
        output_path (str): Path to save the generated plot.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # Group and plot per-agent losses by split rows 0,3,6.../1,4,7.../2,5,8...
    num_agents = 3
    have_step = 'step' in df.columns
    fig, axes = plt.subplots(num_agents, 1, figsize=(10, 5 * num_agents))
    for agent in range(num_agents):
        agent_df = df.iloc[agent::num_agents].reset_index(drop=True)
        if have_step:
            x = agent_df['step']
        else:
            x = range(len(agent_df))
        for col in agent_df.columns:
            if col != 'step':
                axes[agent].plot(x, agent_df[col], label=col)
        axes[agent].set_title(f'Agent {agent+1} Loss Components')
        axes[agent].set_xlabel('Training Step' if have_step else 'Epochs')
        axes[agent].set_ylabel('Loss')
        axes[agent].legend()
        axes[agent].grid(True)

    plt.tight_layout()
    if save:
        path = os.path.join(base_path, output_path)
        plt.savefig(path)
        print(f"Saved losses curve to {path}")
    plt.show()