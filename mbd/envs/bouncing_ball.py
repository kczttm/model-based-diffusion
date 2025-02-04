import os
import jax
from jax import numpy as jnp
import brax
from brax.envs.base import PipelineEnv, State
from brax.generalized import pipeline
from brax.io import html, mjcf

import mbd


class BouncingBall(PipelineEnv):
    """A bouncing ball environment with z-axis force control."""

    def __init__(self, backend: str = "generalized"):
        # Load the XML configuration
        sys = mjcf.load(f"{mbd.__path__[0]}/assets/bouncing_ball.xml")
        super().__init__(sys, backend=backend, n_frames=5)

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to initial state."""
        q = self.sys.init_q  # Initial position (height = 1m)
        qd = jnp.zeros(self.sys.qd_size())  # Initial velocity
        
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        # Initialize reward and done to 0.0 and False
        reward = jnp.array(0.0, dtype=jnp.float32)
        done = jnp.array(0.0, dtype=jnp.float32)
        metrics = {}
        
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        # Clip actions to [-1, 1]
        action = jnp.clip(action, -1.0, 1.0)
        
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(pipeline_state)
        done = self._get_done(pipeline_state)
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        """Returns observation array from state."""
        # Only use position and velocity for observations
        return jnp.concatenate([
            pipeline_state.q,  # z position
            pipeline_state.qd,  # z velocity
        ])

    def _get_reward(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        """Calculate reward based on height maintenance."""
        z_pos = pipeline_state.q[0]
        z_vel = pipeline_state.qd[0]
        
        target_height = 0.5
        height_error = -jnp.abs(z_pos - target_height)
        vel_penalty = -0.1 * z_vel ** 2
        
        # Remove control cost since we can't access it directly
        return height_error + vel_penalty

    def _get_done(self, pipeline_state: pipeline.State) -> jnp.ndarray:
        """Check if episode should terminate."""
        z_pos = pipeline_state.q[0]
        z_vel = pipeline_state.qd[0]
        
        # Done if ball hits ground with high velocity or goes too high
        done = jnp.logical_or(
            jnp.logical_and(z_pos <= 0.01, z_vel < -5.0),
            z_pos > 2.0
        )
        
        return done.astype(jnp.float32)

    @property
    def action_size(self):
        return 1  # z-force

    @property
    def observation_size(self):
        return 2  # [z_pos, z_vel] only


def main():
    """Test the environment with random actions."""
    env = BouncingBall()
    rng = jax.random.PRNGKey(1)
    env_step = jax.jit(env.step)
    env_reset = jax.jit(env.reset)
    
    state = env_reset(rng)
    rollout = [state.pipeline_state]
    
    # Run simulation with random actions
    for _ in range(50):
        rng, rng_act = jax.random.split(rng)
        act = jax.random.uniform(rng_act, (env.action_size,), minval=-1.0, maxval=1.0)
        state = env_step(state, act)
        rollout.append(state.pipeline_state)
    
    # Render and save visualization
    webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)
    path = f"{mbd.__path__[0]}/../results/bouncing_ball"
    if not os.path.exists(path):
        os.makedirs(path)
    html_path = f"{path}/vis.html"
    with open(html_path, "w") as f:
        f.write(webpage)


if __name__ == "__main__":
    main()
    ## TODO: the ball sim is working but not sure about the control, also it penerate the surface.