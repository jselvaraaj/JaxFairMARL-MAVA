from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.mpe.simple import State as SimpleMPEState
from jaxmarl.environments.mpe.simple_spread import SimpleSpreadMPE
from jaxmarl.environments.spaces import Box


@struct.dataclass
class FairSpreadState(SimpleMPEState):
    """State for JaxMarlFairSpread."""

    # Indexed by agent index, the value is the assigned landmark index.
    assignment: Optional[chex.Array] = None
    landmark_occupancy_flag: Optional[chex.Array] = None


class JaxMarlFairSpread(SimpleSpreadMPE):
    """
    JaxMarlFairSpread environment.

    In this environment, agents are assigned to specific landmarks at the beginning of each episode.
    """

    def __init__(
        self,
        num_agents: int = 3,
        num_landmarks: int = 3,
        **kwargs,
    ) -> None:
        """Initialise the environment."""
        assert num_agents == num_landmarks, "Number of agents and landmarks must be equal."

        super().__init__(
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            **kwargs,
        )

        # The new observation space includes the relative position to the assigned landmark (2,).
        new_obs_size = 2 + 2 + 3
        self.observation_spaces = {
            agent: Box(-jnp.inf, jnp.inf, (new_obs_size,)) for agent in self.agents
        }

        self.landmark_occupancy_flag_threshold = 0.5

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], FairSpreadState]:
        """Reset the environment."""
        key_pos, key_assign = jax.random.split(key)
        key_a, key_l = jax.random.split(key_pos)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(key_a, (self.num_agents, 2), minval=-1.0, maxval=1.0),
                jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-1.0, maxval=1.0),
            ]
        )

        assignment = jax.random.permutation(key_assign, jnp.arange(self.num_landmarks))

        state = FairSpreadState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            assignment=assignment,
        )
        state = state.replace(landmark_occupancy_flag=self.get_occupancy_flag(state))

        return self.get_obs(state), state

    def get_obs(self, state: FairSpreadState) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0,))
        def _get_assigned_landmark_info(aidx: int) -> chex.Array:
            assigned_landmark_idx = state.assignment[aidx]
            landmark_pos = state.p_pos[self.num_agents + assigned_landmark_idx]
            relative_pos = landmark_pos - state.p_pos[aidx]
            landmark_occupancy_flag = state.landmark_occupancy_flag[assigned_landmark_idx]
            return relative_pos, landmark_occupancy_flag

        landmark_pos, landmark_occupancy_flag = _get_assigned_landmark_info(self.agent_range)

        def _obs(aidx: int) -> chex.Array:
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 2
                    landmark_occupancy_flag[aidx].flatten(),  # 1
                ]
            )

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: FairSpreadState) -> Dict[str, float]:
        """Get rewards for all agents."""

        # Agent-agent collision penalties
        @partial(jax.vmap, in_axes=(0, None))
        def get_num_collisions(agent_idx: int, other_idx: int) -> chex.Array:
            return jnp.sum(
                jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                    agent_idx,
                    other_idx,
                    state,
                )
            )

        num_collisions = get_num_collisions(self.agent_range, self.agent_range)
        collision_reward = -num_collisions

        # Reward for distance to assigned landmark
        @partial(jax.vmap, in_axes=(0,))
        def _agent_landmark_reward(aidx: int) -> float:
            agent_pos = state.p_pos[aidx]
            assigned_landmark_idx = state.assignment[aidx]
            landmark_pos = state.p_pos[self.num_agents + assigned_landmark_idx]
            return -jnp.linalg.norm(agent_pos - landmark_pos)

        landmark_rewards = _agent_landmark_reward(self.agent_range)

        # Combine rewards
        rewards = {
            agent: collision_reward[i] * self.local_ratio
            + landmark_rewards[i] * (1 - self.local_ratio)
            for i, agent in enumerate(self.agents)
        }
        return rewards

    def get_occupancy_flag(self, state: FairSpreadState) -> chex.Array:
        """Get occupancy flag for each landmark."""
        # Get agent and landmark positions
        agent_pos = state.p_pos[: self.num_agents]  # (num_agents, 2)
        landmark_pos = state.p_pos[self.num_agents :]  # (num_landmarks, 2)

        # Calculate distance matrix: agent-to-landmark distances
        dist_matrix = jnp.linalg.norm(
            agent_pos[:, None, :] - landmark_pos[None, :, :], axis=-1
        )  # (num_agents, num_landmarks)

        normalized_landmark_coverage = (
            jnp.clip(
                jnp.min(dist_matrix, axis=0),  # (num_landmarks,)
                0,
                self.landmark_occupancy_flag_threshold,
            )
            / self.landmark_occupancy_flag_threshold
        )
        occupancy_flag = 1 - normalized_landmark_coverage
        return occupancy_flag

    def step_env(
        self, key: chex.PRNGKey, state: FairSpreadState, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], FairSpreadState, Dict[str, float], Dict[str, bool], Dict]:
        """Step the environment."""
        # The parent step_env will return a SimpleMPEState, so we need to preserve the assignment.
        # This is okay since we don't change assignment during the episode.
        obs, next_mpe_state, rewards, dones, info = super().step_env(key, state, actions)

        # Add type annotation.
        next_mpe_state: SimpleMPEState = next_mpe_state

        # Create a new FairSpreadState, carrying over the assignment from the previous state.
        next_state = FairSpreadState(
            p_pos=next_mpe_state.p_pos,
            p_vel=next_mpe_state.p_vel,
            c=next_mpe_state.c,
            done=next_mpe_state.done,
            step=next_mpe_state.step,
            assignment=state.assignment,
        )
        next_state = next_state.replace(landmark_occupancy_flag=self.get_occupancy_flag(next_state))

        return self.get_obs(next_state), next_state, rewards, dones, info
