
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.env.utils.rendering import render
import torch
import pickle
import argparse
import time


def main(args):
    with open(f"competition_configs/wildfire/{args.config}.pkl", "rb") as f:
        wildfire_configuration = pickle.load(f)

    env = wildfire_v0.parallel_env(
        parallel_envs=args.parallel_envs,
        max_steps=args.max_steps,
        configuration=wildfire_configuration,
        device=torch.device("cpu"),
        buffer_size=args.buffer_size,
        show_bad_actions=False,
        observe_other_power=False,
        observe_other_suppressant=False,
        log_directory=f"test_logging/{args.log}",
        override_initialization_check=True,
        render_mode=args.render_mode,
    )
    env.reset()
    env = action_mapping_wrapper_v0(env)
    observations, infos = env.reset()

    from free_range_zoo.envs.wildfire.agents import (
        MohitoActor,
        MohitoAgent
    )

    from free_range_zoo.envs.wildfire.baselines import (
        RandomBaseline,
        StrongestBaseline
    )


    agents = {
        env.agents[0]: MohitoAgent(agent_name="firefighter_1", parallel_envs=1),
        env.agents[1]: MohitoAgent(agent_name="firefighter_2", parallel_envs=1),
        env.agents[2]: MohitoAgent(agent_name="firefighter_3", parallel_envs=1),
    }

    while not torch.all(env.finished):

        for agent_name, agent in agents.items():
            agent.observe(observations[agent_name])  # Policy observation

        agent_actions = {
            agent_name: agents[agent_name].act(
                action_space=env.action_space(agent_name)
            )
            for agent_name in env.agents
        }  # Policy action determination here

        observations, rewards, terminations, truncations, infos = env.step(
            agent_actions
        )

    render(f"test_logging/{args.log}/0.csv", render_mode=env.render_mode)

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-config",
        type=str,
        default="WS1",
        help="",
    )
    parser.add_argument(
        "-parallel_envs",
        type=int,
        default="1",
        help="",
    )
    parser.add_argument(
        "-max_steps",
        type=int,
        default="100",
        help="",
    )
    parser.add_argument(
        "-buffer_size",
        type=int,
        default="50",
        help="",
    )
    parser.add_argument(
        "-log",
        type=str,
        default=f"{time.time()}",
        help="",
    )
    parser.add_argument(
        "-render_mode",
        type=str,
        default="human",
        help="",
    )

    args = parser.parse_args()

    main(args)
