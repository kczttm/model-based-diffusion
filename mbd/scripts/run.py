import tyro
import numpy as np
from dataclasses import dataclass, field
from typing import List

import mbd


@dataclass
class Args:
    algo: str = "mc_mbd"  # path_integral
    mode: str = "seed"  # temp
    env_name: str = "ant"


def run_multiple_seed(args: Args):
    rews = []
    for seed in range(8):
        if args.algo == "path_integral":
            local_args = mbd.planners.path_integral.Args(
                seed=seed, env_name=args.env_name
            )
            rew = mbd.planners.path_integral.run_path_integral(local_args)
        elif args.algo == "mc_mbd":
            local_args = mbd.planners.mc_mbd.Args(
                seed=seed, env_name=args.env_name, render=False
            )
            rew = mbd.planners.mc_mbd.run_diffusion(local_args)
        else:
            raise NotImplementedError
        rews.append(rew)
    rews = np.array(rews)
    print(f"rew: {rews.mean():.2f} \pm {rews.std():.2f}")


def run_multiple_temp(args: Args):
    temps = np.array([0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.6, 0.8])
    rews = []
    for temp in temps:
        if args.algo == "path_integral":
            local_args = mbd.planners.path_integral.Args(
                seed=0, env_name=args.env_name, temp_sample=temp
            )
            rew = mbd.planners.path_integral.run_path_integral(local_args)
        elif args.algo == "mc_mbd":
            local_args = mbd.planners.mc_mbd.Args(
                seed=0,
                env_name=args.env_name,
                temp_sample=temp,
                render=False,
                disable_recommended_params=True,
            )
            rew = mbd.planners.mc_mbd.run_diffusion(local_args)
        rews.append(rew)
    rews = np.array(rews)
    best_temp = temps[np.argmax(rews)]
    print(f"rews: {rews}")
    print(f"best_temp: {best_temp:.2f}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.mode == "seed":
        run_multiple_seed(args)
    elif args.mode == "temp":
        run_multiple_temp(args)
