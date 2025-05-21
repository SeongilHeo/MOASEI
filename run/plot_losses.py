import pandas as pd
import matplotlib.pyplot as plt
import os

def main(csv_path='losses.csv', output_path='loss_curve_losses.png'):
    """
    Reads a CSV of logged losses and visualizes each loss component over training steps.
    """
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
    plt.savefig(output_path)
    print(f"Saved loss curve to {output_path}")
    plt.show()
    return

if __name__ == '__main__':
    main()
