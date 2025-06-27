"""Basic tests to verify pytest setup."""

import jax
import jax.numpy as jnp
import pytest


def test_jax_import():
    """Test that JAX imports correctly."""
    assert jax is not None
    assert jnp is not None


def test_jax_random():
    """Test that JAX random functions work."""
    key = jax.random.PRNGKey(0)
    random_array = jax.random.normal(key, (3, 3))
    assert random_array.shape == (3, 3)
    assert isinstance(random_array, jnp.ndarray)


def test_mava_import():
    """Test that mava package can be imported."""
    try:
        import mava

        assert mava is not None
    except ImportError:
        pytest.skip("mava package not available")


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works."""
    assert True
