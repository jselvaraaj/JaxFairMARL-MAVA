# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Tuple, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
from brax.envs import State as BraxState
from chex import Array, PRNGKey
from gymnax.environments import spaces as gymnax_spaces
from jaxmarl.environments import SMAX
from jaxmarl.environments import spaces as jaxmarl_spaces
from jaxmarl.environments.mabrax import MABraxEnv
from jaxmarl.environments.mpe.simple import State as MPEState
from jaxmarl.environments.mpe.simple_spread import SimpleSpreadMPE
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import GraphObservation, GraphsTuple, Observation, ObservationGlobalState, State
from mava.wrappers.graph_wrapper import GraphWrapper

# Define a TypeVar for the state, bound to the base State type
JaxMarlStateType = TypeVar("JaxMarlStateType", bound=State)

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class JaxMarlState(Generic[JaxMarlStateType]):
    """Wrapper around a JaxMarl state to provide necessary attributes for jumanji environments."""

    state: JaxMarlStateType
    key: chex.PRNGKey
    step: int


def _is_discrete(space: jaxmarl_spaces.Space) -> bool:
    """JaxMarl sometimes uses gymnax and sometimes uses their own specs, so this is needed."""
    return isinstance(space, (gymnax_spaces.Discrete, jaxmarl_spaces.Discrete))


def _is_box(space: jaxmarl_spaces.Space) -> bool:
    """JaxMarl sometimes uses gymnax and sometimes uses their own specs, so this is needed."""
    return isinstance(space, (gymnax_spaces.Box, jaxmarl_spaces.Box))


def _is_dict(space: jaxmarl_spaces.Space) -> bool:
    """JaxMarl sometimes uses gymnax and sometimes uses their own specs, so this is needed."""
    return isinstance(space, (gymnax_spaces.Dict, jaxmarl_spaces.Dict))


def _is_tuple(space: jaxmarl_spaces.Space) -> bool:
    return isinstance(space, (gymnax_spaces.Tuple, jaxmarl_spaces.Tuple))


def batchify(x: Dict[str, Array], agents: List[str]) -> Array:
    """Stack dictionary values into a single array."""
    return jnp.stack([x[agent] for agent in agents])


def unbatchify(x: Array, agents: List[str]) -> Dict[str, Array]:
    """Split array into dictionary entries."""
    return {agent: x[i] for i, agent in enumerate(agents)}


def merge_space(
    spec: Dict[str, Union[jaxmarl_spaces.Box, jaxmarl_spaces.Discrete]],
) -> jaxmarl_spaces.Space:
    """Convert a dictionary of spaces into a single space with a num_agents size first dimension.

    JaxMarl uses a dictionary of specs, one per agent. For now we want this to be a single spec.
    """
    n_agents = len(spec)
    # Get the first agent's spec from the dictionary.
    single_spec = copy.deepcopy(next(iter(spec.values())))

    err = f"Unsupported space for merging spaces, expected Box or Discrete, got {type(single_spec)}"
    assert _is_discrete(single_spec) or _is_box(single_spec), err

    new_shape = (n_agents, *single_spec.shape)
    single_spec.shape = new_shape

    return single_spec


def is_homogenous(env: MultiAgentEnv) -> bool:
    """Check that all agents in an environment have the same observation and action spaces.

    Note: currently this is done by checking the shape of the observation and action spaces
    as gymnax/jaxmarl environments do not have a custom __eq__ for their specs.
    """
    agents = list(env.observation_spaces.keys())

    main_agent_obs_shape = env.observation_space(agents[0]).shape
    main_agent_act_shape = env.action_space(agents[0]).shape
    # Cannot easily check low, high and n are the same, without being very messy.
    # Unfortunately gymnax/jaxmarl doesn't have a custom __eq__ for their specs.
    same_obs_shape = all(
        env.observation_space(agent).shape == main_agent_obs_shape for agent in agents[1:]
    )
    same_act_shape = all(
        env.action_space(agent).shape == main_agent_act_shape for agent in agents[1:]
    )

    return same_obs_shape and same_act_shape


def jaxmarl_space_to_jumanji_spec(space: jaxmarl_spaces.Space) -> specs.Spec:
    """Convert a jaxmarl space to a jumanji spec."""
    if _is_discrete(space):
        # jaxmarl have multi-discrete, but don't seem to use it.
        if space.shape == ():
            return specs.DiscreteArray(num_values=space.n, dtype=space.dtype)
        else:
            return specs.MultiDiscreteArray(
                num_values=jnp.full(space.shape, space.n), dtype=space.dtype
            )
    elif _is_box(space):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
        )
    elif _is_dict(space):
        # Jumanji needs something to hold the specs
        constructor = namedtuple("SubSpace", list(space.spaces.keys()))  # type: ignore
        # Recursively convert spaces to specs
        sub_specs = {
            sub_space_name: jaxmarl_space_to_jumanji_spec(sub_space)
            for sub_space_name, sub_space in space.spaces.items()
        }
        return specs.Spec(constructor=constructor, name="", **sub_specs)
    elif _is_tuple(space):
        # Jumanji needs something to hold the specs
        field_names = [f"sub_space_{i}" for i in range(len(space.spaces))]
        constructor = namedtuple("SubSpace", field_names)  # type: ignore
        # Recursively convert spaces to specs
        sub_specs = {
            f"sub_space_{i}": jaxmarl_space_to_jumanji_spec(sub_space)
            for i, sub_space in enumerate(space.spaces)
        }
        return specs.Spec(constructor=constructor, name="", **sub_specs)
    else:
        raise ValueError(f"Unsupported JaxMarl space: {space}")


class JaxMarlWrapper(Wrapper, ABC):
    """A wrapper for JaxMarl environments to make their API compatible with Jumanji environments."""

    def __init__(
        self,
        env: MultiAgentEnv,
        has_global_state: bool,
        # We set this to -1 to make it an optional input for children of this class.
        # They must set their own defaults or use the wrapped envs value.
        time_limit: int = -1,
    ) -> None:
        """Initialize the JaxMarlWrapper.

        Args:
        ----
        - env: The JaxMarl environment to wrap.
        - has_global_state: Whether the environment has global state.
        - time_limit: The time limit for each episode.
        """
        # Check that all specs are the same as we only support homogeneous environments, for now ;)
        homogenous_error = (
            f"Mava only supports environments with homogeneous agents, "
            f"but you tried to use {env} which is not homogeneous."
        )
        assert is_homogenous(env), homogenous_error
        # Making sure the child envs set this correctly.
        assert time_limit > 0, f"Time limit must be greater than 0, got {time_limit}"

        self.has_global_state = has_global_state
        self.time_limit = time_limit
        super().__init__(env)
        self._env: MultiAgentEnv
        self.agents = self._env.agents
        self.num_agents = self._env.num_agents

        # Calling these on init to cache the values in a non-jitted context.
        self.state_size  # noqa: B018
        self.action_dim  # noqa: B018

    def reset(
        self, key: PRNGKey
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        key, reset_key = jax.random.split(key)
        obs, env_state = self._env.reset(reset_key)

        metrics: Dict[str, Any] = {"env_metrics": {}}  # default to no metrics
        obs = self._create_observation(obs, env_state)
        state = JaxMarlState(env_state, key, jnp.array(0, dtype=int))
        timestep = restart(obs, shape=(self.num_agents,), extras=metrics)

        return state, timestep

    def step(
        self, state: JaxMarlState, action: Array
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        key, step_key = jax.random.split(state.key)
        obs, env_state, reward, done, _ = self._env.step(
            step_key, state.state, unbatchify(action, self.agents)
        )

        metrics: Dict[str, Any] = {"env_metrics": {}}  # default to no metrics
        obs = self._create_observation(obs, env_state)
        obs = obs._replace(step_count=jnp.repeat(state.step, self.num_agents))
        step_type = jax.lax.select(done["__all__"], StepType.LAST, StepType.MID)

        ts = TimeStep(
            step_type=step_type,
            reward=batchify(reward, self.agents),
            discount=(1.0 - batchify(done, self.agents)).astype(float),
            observation=obs,
            extras=metrics,
        )
        state = JaxMarlState(env_state, key, state.step + jnp.array(1, dtype=int))

        return state, ts

    def _create_observation(
        self,
        obs: Dict[str, Array],
        wrapped_env_state: Any,
    ) -> Union[Observation, ObservationGlobalState]:
        """Create an observation from the raw observation and environment state."""
        obs_data = {
            "agents_view": batchify(obs, self.agents),
            "action_mask": self.action_mask(wrapped_env_state),
            "step_count": jnp.zeros(self.num_agents, dtype=int),
        }
        if self.has_global_state:
            obs_data["global_state"] = self.get_global_state(wrapped_env_state, obs)
            return ObservationGlobalState(**obs_data)

        return Observation(**obs_data)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        agents_view = jaxmarl_space_to_jumanji_spec(merge_space(self._env.observation_spaces))

        action_mask = specs.BoundedArray(
            (self.num_agents, self.action_dim), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray(
            (self.num_agents,), jnp.int32, 0, self.time_limit, "step_count"
        )

        if self.has_global_state:
            global_state = specs.Array(
                (self.num_agents, self.state_size),
                agents_view.dtype,
                "global_state",
            )

            return specs.Spec(
                ObservationGlobalState,
                "ObservationSpec",
                agents_view=agents_view,
                action_mask=action_mask,
                global_state=global_state,
                step_count=step_count,
            )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.Spec:
        return jaxmarl_space_to_jumanji_spec(merge_space(self._env.action_spaces))

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    @property
    def unwrapped(self) -> MultiAgentEnv:
        return self._env

    @abstractmethod
    def action_mask(self, wrapped_env_state: Any) -> Array:
        """Get action mask for each agent."""
        ...

    @abstractmethod
    def get_global_state(self, wrapped_env_state: Any, obs: Dict[str, Array]) -> Array:
        """Get global state from observation for each agent."""
        ...

    @cached_property
    @abstractmethod
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        ...

    @cached_property
    @abstractmethod
    def state_size(self) -> chex.Array:
        """Get the sate size of the global observation"""
        ...


class SmaxWrapper(JaxMarlWrapper):
    """Wrapper for SMAX environment"""

    def __init__(
        self,
        env: MultiAgentEnv,
        has_global_state: bool = False,
    ):
        super().__init__(env, has_global_state, env.max_steps)
        self._env: SMAX

    def reset(
        self, key: PRNGKey
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        state, ts = super().reset(key)
        extras = {"env_metrics": {"won_episode": False}}
        ts = ts.replace(extras=extras)
        return state, ts

    def step(
        self, state: JaxMarlState, action: Array
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        state, ts = super().step(state, action)

        current_winner = (ts.step_type == StepType.LAST) & jnp.all(ts.reward >= 1.0)
        extras = {"env_metrics": {"won_episode": current_winner}}
        ts = ts.replace(extras=extras)
        return state, ts

    @cached_property
    def state_size(self) -> chex.Array:
        """Get the sate size of the global observation"""
        return self._env.state_size

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        single_agent_action_space = self._env.action_space(self.agents[0])
        return single_agent_action_space.n

    def action_mask(self, wrapped_env_state: Any) -> Array:
        """Get action mask for each agent."""
        avail_actions = self._env.get_avail_actions(wrapped_env_state)
        return jnp.array(batchify(avail_actions, self.agents), dtype=bool)

    def get_global_state(self, wrapped_env_state: Any, obs: Dict[str, Array]) -> Array:
        """Get global state from observation and copy it for each agent."""
        return jnp.tile(jnp.array(obs["world_state"]), (self.num_agents, 1))


class MabraxWrapper(JaxMarlWrapper):
    """Wrraper for the Mabrax environment."""

    def __init__(
        self,
        env: MABraxEnv,
        has_global_state: bool = False,
    ):
        super().__init__(env, has_global_state, env.episode_length)
        self._env: MABraxEnv

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        return self._env.action_space(self.agents[0]).shape[0]

    @cached_property
    def state_size(self) -> chex.Array:
        """Get the sate size of the global observation"""
        brax_env = self._env.env
        return brax_env.observation_size

    def action_mask(self, wrapped_env_state: BraxState) -> Array:
        """Get action mask for each agent."""
        return jnp.ones((self.num_agents, self.action_dim), dtype=bool)

    def get_global_state(self, wrapped_env_state: BraxState, obs: Dict[str, Array]) -> Array:
        """Get global state from observation and copy it for each agent."""
        # Use the global state of brax.
        return jnp.tile(wrapped_env_state.obs, (self.num_agents, 1))


class MPEWrapper(JaxMarlWrapper):
    """Wrapper for the MPE environment."""

    def __init__(
        self,
        env: SimpleSpreadMPE,
        has_global_state: bool = False,
    ):
        super().__init__(env, has_global_state, env.max_steps)
        self._env: SimpleSpreadMPE

    @cached_property
    def action_dim(self) -> chex.Array:
        "Get the actions dim for each agent."
        # Adjusted automatically based on the action_type specified in the kwargs.
        if _is_discrete(self._env.action_space(self.agents[0])):
            return self._env.action_space(self.agents[0]).n
        return self._env.action_space(self.agents[0]).shape[0]

    @cached_property
    def state_size(self) -> chex.Array:
        "Get the state size of the global observation"
        return self._env.observation_space(self.agents[0]).shape[0] * self.num_agents

    def action_mask(self, wrapped_env_state: Any) -> Array:
        """Get action mask for each agent."""
        return jnp.ones((self.num_agents, self.action_dim), dtype=bool)

    def get_global_state(self, wrapped_env_state: Any, obs: Dict[str, Array]) -> Array:
        """Get global state from observation and copy it for each agent."""
        global_state = jnp.concatenate([obs[agent_id] for agent_id in obs])
        return jnp.tile(global_state, (self.num_agents, 1))


class MPEGraphWrapper(GraphWrapper):
    """Wrapper for the MPE environment that adds a graph to the observation.

    This wrapper creates a graph topology for each agent where:
    - Each agent and landmark is represented as a node in the graph
    - Node features are relative positions and velocities with respect to the ego agent
      (4D features: [relative_x, relative_y, relative_vx, relative_vy])
    - Edges are created based on a visibility radius - nodes are connected only if they
      are within this radius of each other
    - Edge features are the Euclidean distances between connected nodes
    - Self-loops can be optionally added to each node

    For example, in a 3-agent environment with 2 landmarks:
    - Each agent gets its own graph with 5 nodes (3 agents + 2 landmarks)
    - For Agent 0's graph:
      * Node features are positions/velocities relative to Agent 0
      * Edges connect nodes that are within visibility_radius of each other
      * Edge features are the distances between connected nodes
      * ego_node_index=0 identifies Agent 0 as the reference point
    - For Agent 1's graph:
      * Node features are positions/velocities relative to Agent 1
      * Different edge connections based on Agent 1's visibility
      * Edge features are the distances between connected nodes
      * ego_node_index=1 identifies Agent 1 as the reference point

    This relative representation allows each agent to have its own perspective of the
    environment, with node features and graph topology specific to its viewpoint.
    """

    def __init__(
        self,
        env: MPEWrapper,
        add_self_loops: bool = True,
        visibility_radius: float = 1,
    ):
        super().__init__(env)
        self._env: MPEWrapper

        self.add_self_loops = add_self_loops
        self.visibility_radius = visibility_radius

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

        self.num_entities = self._env.num_entities
        self.node_features_dim = 4

    def visibility_graph_for_ego(
        self,
        state: MPEState,
        visibility_radius: float,
        ego_idx: int,
    ) -> GraphsTuple:
        """Return a GraphsTuple for ONE ego agent, with edges defined by a
        global, uniform visibility radius."""

        positions = state.p_pos

        dists = jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)

        mask = dists <= visibility_radius
        if not self.add_self_loops:
            mask = mask.at[jnp.arange(self.num_entities), jnp.arange(self.num_entities)].set(False)

        max_n_edge = self.num_entities * self.num_entities
        senders, receivers = jnp.nonzero(mask, size=max_n_edge, fill_value=-1)

        # ------------------------------------------------------------------
        # build a "safe" distance matrix
        # *shape* = (N+1, N+1) so index N is guaranteed valid
        # last row / col are all zeros
        # ------------------------------------------------------------------
        safe_dists = jnp.pad(  # (N, N)  ->  (N+1, N+1)
            dists,
            pad_width=((0, 1), (0, 1)),
            mode="constant",
            constant_values=0.0,
        )
        N = self.num_entities
        safe_senders = jnp.where(senders < 0, N, senders)
        safe_receivers = jnp.where(receivers < 0, N, receivers)
        # for invalid edges, edge feature would be 0.0
        edge_features = safe_dists[safe_senders, safe_receivers][..., None]

        node_features = jnp.concatenate(
            [positions - positions[ego_idx], state.p_vel - state.p_vel[ego_idx]], axis=-1
        )
        assert node_features.shape[-1] == self.node_features_dim, (
            f"Node features dim specified in MPEWrapper is {self.node_features_dim}, "
            f"but got {node_features.shape[-1]} for agent {ego_idx}."
        )

        n_node = jnp.asarray([self.num_entities])
        n_edge = jnp.asarray([max_n_edge])

        return GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=None,
            ego_node_index=jnp.asarray([ego_idx]),
        )

    def add_graph_to_observations(
        self, state: JaxMarlState[MPEState], observation: Union[Observation, ObservationGlobalState]
    ) -> GraphObservation:
        b_graph = jax.vmap(self.visibility_graph_for_ego, in_axes=(None, None, 0))(
            state.state, self.visibility_radius, jnp.arange(self.num_agents)
        )
        return GraphObservation(observation=observation, graph=b_graph)

    @cached_property
    def observation_spec(
        self,
    ) -> Union[
        specs.Spec[GraphObservation[Observation]],
        specs.Spec[GraphObservation[ObservationGlobalState]],
    ]:
        """Define the observation spec for the Jraph graph representation."""
        obs_spec = self._env.observation_spec

        max_n_edge = self.num_entities * self.num_entities

        graph_spec = specs.Spec(
            constructor=GraphsTuple,
            name="graph",
            nodes=specs.Array(
                shape=(
                    self.num_agents,
                    self.num_entities,
                    self.node_features_dim,
                ),
                dtype=jnp.float32,
                name="nodes",
            ),
            edges=specs.Array(
                shape=(self.num_agents, max_n_edge, 1), dtype=jnp.float32, name="edges"
            ),
            senders=specs.Array(
                shape=(self.num_agents, max_n_edge), dtype=jnp.int32, name="senders"
            ),
            receivers=specs.Array(
                shape=(self.num_agents, max_n_edge), dtype=jnp.int32, name="receivers"
            ),
            n_node=specs.Array(shape=(self.num_agents, 1), dtype=jnp.int32, name="n_node"),
            n_edge=specs.Array(shape=(self.num_agents, 1), dtype=jnp.int32, name="n_edge"),
            globals=None,
            ego_node_index=specs.Array(
                shape=(self.num_agents, 1), dtype=jnp.int32, name="ego_node_index"
            ),
        )

        return specs.Spec(
            GraphObservation,
            "GraphObservation",
            observation=obs_spec,
            graph=graph_spec,
        )
