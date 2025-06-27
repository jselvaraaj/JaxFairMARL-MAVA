import jax
import jax.numpy as jnp
import pytest

from mava.environments.jaxmarlfairspread import FairSpreadState, JaxMarlFairSpread


@pytest.fixture
def env() -> JaxMarlFairSpread:
    """Fixture for JaxMarlFairSpread environment."""
    return JaxMarlFairSpread(num_agents=3, num_landmarks=3)


def test_reset_and_step(env: JaxMarlFairSpread) -> None:
    """Test the reset and step functions."""
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    # Test reset
    assert isinstance(state, FairSpreadState)
    assert state.step == 0
    assert state.assignment.shape == (env.num_agents,)
    # Check if assignment is a permutation
    assert jnp.array_equal(jnp.sort(state.assignment), jnp.arange(env.num_landmarks))

    # Check obs shapes from reset
    expected_obs_shape = env.observation_space(env.agents[0]).shape
    for agent in env.agents:
        assert obs[agent].shape == expected_obs_shape

    # Test step
    key, action_key, step_key = jax.random.split(key, 3)
    actions = {agent: env.action_space(agent).sample(action_key) for agent in env.agents}
    obs, next_state, rewards, dones, info = env.step_env(step_key, state, actions)

    assert isinstance(next_state, FairSpreadState)
    assert next_state.step == state.step + 1

    # Check output shapes from step
    for agent in env.agents:
        assert obs[agent].shape == expected_obs_shape
        assert isinstance(rewards[agent], jnp.ndarray)
        assert isinstance(dones[agent], jnp.ndarray)


def test_rewards(env: JaxMarlFairSpread) -> None:
    """Test the reward function."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Place agent 0 at its assigned landmark to make its landmark reward 0
    assigned_landmark_idx = state.assignment[0]
    landmark_pos = state.p_pos[env.num_agents + assigned_landmark_idx]
    state = state.replace(p_pos=state.p_pos.at[0].set(landmark_pos))

    # Place other agents far away to avoid collisions.
    state = state.replace(p_pos=state.p_pos.at[1].set(jnp.array([10.0, 10.0])))
    state = state.replace(p_pos=state.p_pos.at[2].set(jnp.array([-10.0, -10.0])))

    rewards = env.rewards(state)

    # Agent 0 should have a reward of 0, since collision reward is 0 and landmark distance is 0.
    assert jnp.isclose(rewards["agent_0"], 0.0)

    # Other agents should have negative rewards due to distance from their landmarks.
    assert rewards["agent_1"] < 0
    assert rewards["agent_2"] < 0
