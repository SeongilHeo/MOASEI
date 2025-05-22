import torch
import argparse


from core import COMACritic, GNNActor, Incidence_graph
from utils import load_configs, init_batch, discount_cumsum, build_joint_action_tensor, save_model, plot_reward_curve, plot_loss_curve

def train():
    """
    Train the A2C model using the specified environment configurations.
    """
    args = handle_args()


    ENVS = load_configs()

    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    num_episodes = args.num_episodes
    batch_size = args.batch_size
    curriculum = args.curriculum

    num_envs = len(ENVS)
    num_agents = 3
    action_dim = 6 


    shared_actor = GNNActor(input_dim=input_dim, hidden_dim=hidden_dim)
    actor_optimizer = torch.optim.Adam(shared_actor.parameters(), lr=1e-2)

    central_critic = COMACritic(input_dim=4, hidden_dim=hidden_dim, action_dim=action_dim+1, num_agents=num_agents)
    critic_optimizer = torch.optim.Adam(central_critic.parameters(), lr=1e-2)

    Graph  = Incidence_graph(node_size=input_dim, all_node=False)

    stores={'reward':[], 'actor_loss':[], 'critic_loss':[]}


    for epoch in range(num_episodes):
        if curriculum:
            env = ENVS[epoch // 50]
        else:
            env = ENVS[epoch % num_envs]

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

            observations, reward, _, _, _ = env.step(env_actions)

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

        for _ in range(8):
            critic_optimizer.zero_grad()
            critic_loss=central_critic.compute_loss(batch_joint_data, batch_joint_actions, batch_joint_rewards)
            critic_loss.backward()
            critic_optimizer.step()

        print(f"Epoch: {epoch}, Reward: {sum(store).item()}")

        stores['reward'].append(sum(store).item())
        stores['actor_loss'].append(total_loss.item())
        stores['critic_loss'].append(critic_loss.item())


    plot_reward_curve(stores['reward'])
    plot_loss_curve(stores)

    save_model(shared_actor, "_a2c")

def handle_args() -> argparse.Namespace:
    """
    Handle script arguments.

    Returns:
        argparse.Namespace - parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate trained policies on a given wildfire configuration.')

    parser.add_argument('input_dim', type=int, default=4, help='input dimension for the model')    
    parser.add_argument('hidden_dim', type=int, default=64, help='hidden dimension for the model')
    parser.add_argument('num_episodes', type=int, default=150, help='number of episodes to train')
    parser.add_argument('batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('curriculum', type=bool, default=False, help='curriculum learning flag')

    return parser.parse_args()

if __name__ == "__main__":
    train()
    import argparse