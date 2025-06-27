"""Shared pytest configuration and fixtures."""

import jax
import pytest


@pytest.fixture(autouse=True)
def set_jax_seed():
    """Set a fixed JAX seed for reproducible tests."""
    jax.config.update("jax_default_prng_impl", "rbg")
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing."""
    # Enable float64 for more precise testing
    jax.config.update("jax_enable_x64", True)
    # Use CPU for consistent testing
    jax.config.update("jax_platform_name", "cpu")
