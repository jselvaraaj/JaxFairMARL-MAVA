import jax.numpy as jnp
from einops import rearrange

from mava.environments.jaxmarlfairspread import FairSpreadState
from mava.types import GraphsTuple
from mava.wrappers.jaxmarl import MPEGraphWrapper, MPEWrapper


class FairMPEAssignmentGraphWrapper(MPEGraphWrapper):
    def __init__(
        self,
        env: MPEWrapper,
        add_self_loops: bool = True,
        visibility_radius: float = 1,
    ):
        super().__init__(env, add_self_loops, visibility_radius)

        self.node_features_dim = 2 + 2 + 1 + 4 + 2

    def visibility_graph_for_ego(
        self,
        state: FairSpreadState,
        visibility_radius: float,
        ego_idx: int,
    ) -> GraphsTuple:
        """Return a GraphsTuple for ONE ego agent, with edges defined by a
        global, uniform visibility radius."""

        max_n_edge = self.num_entities * self.num_entities

        dists = jnp.linalg.norm(state.p_pos[:, None, :] - state.p_pos[None, :, :], axis=-1)

        mask = dists <= visibility_radius
        if not self.add_self_loops:
            mask = mask.at[jnp.arange(self.num_entities), jnp.arange(self.num_entities)].set(False)

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

        nearest_landmark_idx = self.num_agents + state.nearest_landmark_idx

        agent_node_features = jnp.concatenate(
            [
                state.p_pos[state.assignment] - state.p_pos[ego_idx][None],
                state.p_vel[state.assignment] - state.p_vel[ego_idx][None],
                state.landmark_occupancy_flag[state.assignment][..., None],
                rearrange(
                    state.p_pos[nearest_landmark_idx] - state.p_pos[ego_idx][None, None],
                    "agents goals coords -> agents (goals coords)",
                ),
                state.landmark_occupancy_flag[nearest_landmark_idx],
            ],
            axis=-1,
        )
        landmark_node_features = jnp.zeros_like(agent_node_features)

        node_features = jnp.concatenate([agent_node_features, landmark_node_features], axis=-2)
        assert node_features.shape[-1] == self.node_features_dim, (
            f"Node features dim specified in FairMPEAssignmentGraphWrapper is "
            f"{self.node_features_dim}, but got {node_features.shape[-1]} for agent {ego_idx}."
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
