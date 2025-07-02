from enum import Enum
from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.mpe.simple import State as SimpleMPEState
from jaxmarl.environments.mpe.simple_spread import SimpleSpreadMPE
from jaxmarl.environments.spaces import Box


class AssignmentStrategy(Enum):
    """Assignment strategy for JaxMarlFairSpread."""

    RANDOM = "random"
    OPTIMAL = "optimal"
    FAIR = "fair"


@struct.dataclass
class FairSpreadState(SimpleMPEState):
    """State for JaxMarlFairSpread."""

    # Indexed by agent index, the value is the assigned landmark index.
    assignment: Optional[chex.Array] = None
    landmark_occupancy_flag: Optional[chex.Array] = None
    nearest_landmark_idx: Optional[chex.Array] = None


class JaxMarlFairSpread(SimpleSpreadMPE):
    """
    JaxMarlFairSpread environment.

    In this environment, agents are assigned to specific landmarks at the beginning of each episode.
    """

    def __init__(
        self,
        num_agents: int = 3,
        num_landmarks: int = 3,
        assignment_strategy: AssignmentStrategy = AssignmentStrategy.RANDOM,
        **kwargs,
    ) -> None:
        """Initialise the environment."""
        assert num_agents == num_landmarks, "Number of agents and landmarks must be equal."

        super().__init__(
            num_agents=num_agents,
            num_landmarks=num_landmarks,
            **kwargs,
        )

        # The new observation space includes:
        # velocity (2) + position (2) + assigned landmark info (3) + two nearest landmarks info (6)
        new_obs_size = 2 + 2 + 6
        self.observation_spaces = {
            agent: Box(-jnp.inf, jnp.inf, (new_obs_size,)) for agent in self.agents
        }

        self.landmark_occupancy_flag_threshold = 0.5
        self.assignment_strategy = assignment_strategy

    def get_new_assignment(self, key: chex.PRNGKey, state: FairSpreadState) -> chex.Array:
        """Get assignment for each agent."""
        if self.assignment_strategy == AssignmentStrategy.RANDOM:
            assignment = jax.random.permutation(key, jnp.arange(self.num_landmarks))
        elif self.assignment_strategy == AssignmentStrategy.OPTIMAL:
            pass
        elif self.assignment_strategy == AssignmentStrategy.FAIR:
            pass
        return assignment

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

        state = FairSpreadState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
        )
        assignment = self.get_new_assignment(key_assign, state)
        state = state.replace(
            assignment=assignment,
            landmark_occupancy_flag=self.get_occupancy_flag(state),
            nearest_landmark_idx=self.get_two_nearest_landmarks_idx(state),
        )

        return self.get_obs(state), state

    def get_two_nearest_landmarks_idx(self, state: FairSpreadState) -> chex.Array:
        @partial(jax.vmap, in_axes=(None, 0))
        def _get_two_nearest_landmarks_idx(state: FairSpreadState, aidx: int) -> chex.Array:
            agent_pos = state.p_pos[aidx]
            landmark_positions = state.p_pos[self.num_agents :]  # All landmark positions

            # Compute distances to all landmarks
            distances = jnp.linalg.norm(landmark_positions - agent_pos, axis=1)

            # Get indices of two nearest landmarks
            nearest_indices = jnp.argsort(distances)[:2]

            return nearest_indices

        nearest_landmark_idx = _get_two_nearest_landmarks_idx(state, self.agent_range)
        return nearest_landmark_idx  # (num_agents, 2)

    def get_obs(self, state: FairSpreadState) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0,))
        def _get_assigned_landmark_info(aidx: int) -> chex.Array:
            assigned_landmark_idx = state.assignment[aidx]
            landmark_pos = state.p_pos[self.num_agents + assigned_landmark_idx]
            relative_pos = landmark_pos - state.p_pos[aidx]
            landmark_occupancy_flag = state.landmark_occupancy_flag[assigned_landmark_idx]
            return relative_pos, landmark_occupancy_flag

        assigned_landmark_pos, assigned_landmark_occupancy_flag = _get_assigned_landmark_info(
            self.agent_range
        )

        landmark_positions = state.p_pos[self.num_agents :]  # All landmark positions

        nearest_landmark_idx = self.get_two_nearest_landmarks_idx(state)

        # Get relative positions and occupancy flags for the two nearest landmarks
        nearest_landmark_pos = (
            landmark_positions[nearest_landmark_idx] - state.p_pos[self.agent_range][..., None, :]
        )
        nearest_landmark_flags = state.landmark_occupancy_flag[nearest_landmark_idx]

        def _obs(aidx: int) -> chex.Array:
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    nearest_landmark_pos[aidx].flatten(),  # 4 (2 landmarks × 2 coords)
                    nearest_landmark_flags[aidx].flatten(),  # 2 (2 landmarks × 1 flag)
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

        if self.assignment_strategy != AssignmentStrategy.RANDOM:
            assignment = self.get_new_assignment(key, next_state)
        else:
            assignment = next_state.assignment

        next_state = next_state.replace(
            assignment=assignment,
            landmark_occupancy_flag=self.get_occupancy_flag(next_state),
            nearest_landmark_idx=self.get_two_nearest_landmarks_idx(next_state),
        )

        return self.get_obs(next_state), next_state, rewards, dones, info
