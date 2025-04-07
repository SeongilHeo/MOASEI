import os
from typing import List, Dict, Any
import random
import free_range_rust
from free_range_zoo.utils.agent import Agent
from free_range_zoo.envs.wildfire.env.utils.in_range_check import chebyshev

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

def build_incidence_graph(self_obs, other_obs, task_obs, node_size = 4):
    """
    self_obs: (4,) tensor
    other_obs: (2, 2) or (2,4) tensor
    task_obs: (N, 4) tensor, N ≤ 6
    """
    N = task_obs.shape[0]  # number of observed tasks
    node_features = []
    edge_index = [[], []]  # 2 × num_edges

    # Node agent: self
    padding_size = node_size - self_obs.shape[0]
    node_features.append(torch.cat([self_obs, torch.zeros(padding_size)]))  # pad to 16

    # Node agent: others
    padding_size = node_size - other_obs.shape[1]
    node_features.append(torch.cat([other_obs[0], torch.zeros(padding_size)]))  # pad to 16
    node_features.append(torch.cat([other_obs[1], torch.zeros(padding_size)]))  # pad to 16

    node_features.sort(key=lambda x: x[0].item())  # sort by y-coordinates 

    # Nodes task: 1 ~ N
    padding_size = node_size - task_obs.shape[1]
    for t in range(N):
        node_features.append(torch.cat([task_obs[t], torch.zeros(padding_size)]))

    # Hyperedge nodes: N+1 ~ N+N
    for task_idx, t in enumerate(task_obs):
        for agent_idx in range(3):
            # Check if agent can reach the task
            agent_pos = node_features[agent_idx][:2]  # agent position
            if chebyshev(agent_pos.unsqueeze(0) , t[:2].unsqueeze(0) , torch.tensor([1])):

                # dummy features for hyperedge node (can be all zeros)
                node_features.append(torch.zeros(node_size))
                hyp_idx = len(node_features) - 1  # hyperedge is last added node

                # Agent — Hyperedge
                edge_index[0].append(agent_idx)
                edge_index[1].append(hyp_idx)
    
                # Task — Hyperedge
                task_node_idx = 3 + task_idx
                edge_index[0].append(task_node_idx)
                edge_index[1].append(hyp_idx)

    x = torch.stack(node_features)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ==== Actor Network ====
class ActorGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ActorGNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(GATConv(in_dim, hidden_dim, heads=2, concat=True))
        for i in range(18):
            self.hidden_layers.append(GATConv(hidden_dim*2, hidden_dim, heads=2, concat=True))
        self.final_gat = GATConv(2 * hidden_dim, in_dim, heads=1, concat=False)

        self.proj = nn.Linear(in_dim, 1)   

    def forward(self, x, edge_index, hyperedge_mask, training=False):
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            if training:
                x = F.dropout(x, p=0.3, training=self.training)

        x = self.final_gat(x, edge_index)

        hyperedge_logits = self.proj(x[hyperedge_mask]).squeeze(-1)

        probs = F.softmax(hyperedge_logits, dim=0)

        max_edge_index = torch.argmax(probs).item()
        
        return x[hyperedge_mask], max_edge_index

# ==== Critic Network ====
class CriticGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, hyperedge_dim=3):
        super(CriticGNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(GATConv(in_dim, hidden_dim, heads=2, concat=True))
        for i in range(18):
            self.hidden_layers.append(GATConv(2*hidden_dim, hidden_dim, heads=2, concat=True))
        self.final_gat = GATConv(2 * hidden_dim, in_dim, heads=1, concat=False)

        self.out_layer = nn.Linear(in_dim + hyperedge_dim*in_dim, 1)

    def forward(self, x, edge_index, hyperedge_edges, batch=None):
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index))
        
        x = self.final_gat(x, edge_index)
        # if batch:
        x = global_mean_pool(x,batch=None)

        x= torch.cat([x,hyperedge_edges],dim=0).view(-1)
        q_value = self.out_layer(x)

        return q_value

# ==== Actor Agent ====
class MohitoActor(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        in_dim = kwargs.get('in_dim', 4)
        hidden_dim = kwargs.get('hidden_dim', 8)
        model_path = kwargs.get('model_path', None)
        device = kwargs.get('device', 'cpu')

        self.actor_graph = ActorGNN(in_dim, hidden_dim)
        self.critic_graph = CriticGNN(in_dim, hidden_dim)

        if model_path and os.path.exists(model_path):
            self.actor_graph.load_state_dict(torch.load(model_path, map_location=device))

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

    def act(self, action_space: free_range_rust.Space) -> List[List[int]]:
        """
        Return a list of actions, one for each parallel environment.

        Args:
            action_space: free_range_rust.Space - Current action space available to the agent.
        Returns:
            List[List[int]] - List of actions, one for each parallel environment.
        """
        return self.actions

    def observe(self, observation: Dict[str, Any], epsilon=0) -> None:
        """
        Observe the environment.

        Args:
            observation: Dict[str, Any] - Current observation from the environment.
        """

        self.observation, self.t_mapping = observation
        self.t_mapping = self.t_mapping['agent_action_mapping']

        has_suppressant = self.observation['self'][:, 3] != 0

        for batch in range(self.parallel_envs):
            self_obs = self.observation['self'][batch]              # shape (4,)
            other_obs = self.observation['others'][batch]           # shape (2, 2)
            task_obs = self.observation['tasks'][batch]             # shape (N, 4)

            num_task = task_obs.shape[0] 
            if num_task == 0:
                self.actions[batch, :] = -1
                continue
            
            inc_graph = build_incidence_graph(self_obs, other_obs, task_obs, node_size=4)

            hyperedge_mask = (torch.arange(inc_graph.x.size(0)) >= 3+num_task)  # Assuming hyperedges start from index 3
            hyperedges, max_edge_index = self.actor_graph.forward(inc_graph.x, inc_graph.edge_index, hyperedge_mask)

            if random.random() < epsilon+1:
                max_edge_index = random.choice(self.t_mapping[0]).item()
                
            hyperedge = hyperedges[max_edge_index, :]

            target_node = max_edge_index+3+num_task 

            # edge_index
            src = inc_graph.edge_index[0]
            dst = inc_graph.edge_index[1]

            # search task node connected to hyperedge
            task_node_indices = src[(dst == target_node) & (src > 2)]
            assert len(task_node_indices) == 1, "There should be exactly one task node connected to the hyperedge"

            action = task_node_indices.item() - 3

            self.actions[batch, 0] = action

        self.actions[:, 1].masked_fill_(~has_suppressant, -1)  # Agents that do not have suppressant noop

        return hyperedge
    
    def evalutate_critic(self, inc_graph, hyperedges):
        """
        Evaluate the critic network.

        Args:
            graph: The graph data containing node features and edge indices.
            edges: The hyperedge features.

        Returns:
            The Q-value for the given edges.
        """
        return self.critic_graph.forward(inc_graph.x, inc_graph.edge_index, hyperedges)