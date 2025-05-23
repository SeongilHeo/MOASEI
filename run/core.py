import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch.distributions import Categorical

from typing import Dict, Any, Tuple
import numpy as np
import os, csv

from free_range_zoo.utils.agent import Agent
import free_range_rust

LOSS_CSV_PATH = os.path.join(os.getcwd(), "losses.csv")

class Incidence_graph:
    def __init__(self, node_size=4, all_node=True):
        """
        Initialize the incidence graph builder.

        Args:
            node_size (int): dimensionality of node feature vectors.
            all_node (bool): whether to include all other-agent nodes.
        """
        self.node_size = node_size

        self.node_features = []
        self.edge_index = [[], []]

        self.self_idx = 0
        self.noop_idx = 1
        self.task_indices = []
        self.otheragent_indices = []
        self.hyperedge_mask = []
        self.otheragent_hyperedge_mask = []
        self.agents_indices = []

        self.all_node = all_node

    def _get_next_node_idx(self):
        """
        Get the next node index for adding a new node.
        """
        return len(self.node_features)
    
    def _add_node(self, feature=None, type=None):
        """
        Add a node to the graph, padding or zero-initializing features, and tag by type.

        Args:
            feature (Tensor or None): initial feature vector for the node.
            type (str): category label for the node (e.g., 'self', 'task').
        """
        # pad or zero-initialize the feature vector
        if feature is None:
            feature = torch.zeros(self.node_size)
        else:
            pad = torch.zeros(self.node_size - feature.size(0))
            feature = torch.cat([feature, pad], dim=0)

        self.node_features.append(feature)
        
        # add node to the appropriate list based on type
        node_idx = len(self.node_features) - 1
        if type == "self":
            self.self_idx = node_idx
        elif type == "noop":
            self.noop_idx = node_idx
        elif type == "task":
            self.task_indices.append(node_idx)
        elif type == "other":
            self.otheragent_indices.append(node_idx)
        elif type == "hyperedge":
            self.hyperedge_mask.append(node_idx)
        elif type == "otheragent_hyperedge":
            self.otheragent_hyperedge_mask.append(node_idx)
        elif type == "agents":
            self.agents_indices.append(node_idx)

    def _add_edge(self, src0, dst, src1=None):
        """
        Add directed edge(s) from source(s) to destination node.
        Supports binary hyperedges when two sources provided.

        Args:
            src0 (int): source node index.
            dst (int): destination node index.
            src1 (int or None): optional second source node index for hyperedges.
        """
        self.edge_index[0].append(src0)
        self.edge_index[1].append(dst)
        if src1 is not None:
            self.edge_index[0].append(src1)
            self.edge_index[1].append(dst)


    def build(self, self_obs, other_obs, task_obs, t_map):
        """
        Construct the incidence graph for a single agent.
        Maps observations to nodes and connects hyperedges for valid actions.

        Args:
            self_obs (Tensor): observation vector for the agent.
            other_obs (Tensor): observation vectors for other agents.
            task_obs (Tensor): observation vectors for tasks.
            t_map (Tensor): mapping of valid tasks to indices.
        Returns:
            Data: PyG Data object containing node features and edge indices.
            hyperedge_mask (Tensor): mask for hyperedge nodes.
            otheragent_indices (Tensor): indices for other-agent nodes.
            otheragent_hyperedge_mask (Tensor): mask for other-agent hyperedges.
        """
        # reset internal buffers
        self.node_features = []
        self.edge_index = [[], []]
        self.task_indices = []
        self.otheragent_indices = []
        self.hyperedge_mask = []
        self.otheragent_hyperedge_mask = []

        num_tasks = task_obs.size(0)
        num_others = other_obs.size(0)

        # 1) Self node
        self._add_node(self_obs, type="self")

        # 2) NOOP node
        self._add_node(type="noop")

        # 3) Task nodes
        for t in range(num_tasks):
            self._add_node(task_obs[t], type="task")

        # 4) Other‐agent nodes
        if self.all_node:
            for o in range(num_others):
                self._add_node(other_obs[o], type="other")
                # connect self → other [?]
                self._add_edge(self.self_idx, self.otheragent_indices[o])

        # 5) Hyperedges: self → NOOP + valid tasks
        # self → NOOP hyperedge
        h_idx = self._get_next_node_idx()
        self._add_node(type="hyperedge")
        self._add_edge(self.self_idx, h_idx, self.noop_idx)
        # self → each valid task
        valid_tasks = t_map.tolist()
        for t in valid_tasks:
            h_idx = self._get_next_node_idx()
            self._add_node(type="hyperedge")
            self._add_edge(self.self_idx, h_idx, self.task_indices[t])

        if self.all_node:
            # 6) Hyperedges: each other‐agent → NOOP + all tasks
            for other_idx in self.otheragent_indices:
                # other → NOOP hyperedge
                h_idx = self._get_next_node_idx()
                self._add_node(type="otheragent_hyperedge")
                self._add_edge(other_idx, h_idx, self.noop_idx)
                # other → each task hyperedge
                for task_idx in self.task_indices:
                    h_idx = self._get_next_node_idx()
                    self._add_node(type="otheragent_hyperedge")
                    self._add_edge(other_idx, h_idx, task_idx)

        # pack into PyG Data
        x = torch.stack(self.node_features)
        edge_index = torch.tensor(
            self.edge_index, 
            dtype=torch.long
        )
        hyperedge_mask = torch.tensor(
            self.hyperedge_mask, 
            dtype=torch.long
        )
        otheragent_indices = torch.tensor(
            self.otheragent_indices, 
            dtype=torch.long
        )
        otheragent_hyperedge_mask = torch.tensor(
            self.otheragent_hyperedge_mask,
            dtype=torch.long
        )

        return (
            Data(x=x, edge_index=edge_index),
            hyperedge_mask,
            otheragent_indices,
            otheragent_hyperedge_mask
        )
    
    def build_joint(self, all_self_obs, task_obs, all_t_map):
        """
        Construct a joint incidence graph for multiple agents.
        Each agent node connects to shared tasks via hyperedges.

        Args:
            all_self_obs (List[Tensor]): list of observation vectors for each agent.
            task_obs (Tensor): observation vectors for tasks.
            all_t_map (List[Tensor]): mapping of valid tasks to indices for each agent.
        Returns:
            Data: PyG Data object containing node features and edge indices.
        """
        # reset internal buffers
        self.node_features = []
        self.edge_index = [[], []]
        self.task_indices = []
        self.agents_indices = []

        num_tasks = task_obs.size(0)
        num_agents = len(all_self_obs)

        # 1) Agents node
        for a in range(num_agents):
            self._add_node(all_self_obs[a], type="agents")

        # 2) NOOP node
        self._add_node(type="noop")

        # 3) Task nodes
        for t in range(num_tasks):
            self._add_node(task_obs[t], type="task")

        # 5) Hyperedges: agents → NOOP + valid tasks
        for a in range(num_agents):
            # agent → NOOP hyperedge
            h_idx = self._get_next_node_idx()
            self._add_node(type="hyperedge")
            self._add_edge(self.agents_indices[a], h_idx, self.noop_idx)
            
            # agent → each task hyperedge
            valid_tasks = all_t_map[a].tolist()
            for t in valid_tasks:
                h_idx = self._get_next_node_idx()
                self._add_node(type="hyperedge")
                self._add_edge(self.agents_indices[a], h_idx, self.task_indices[t])

        # pack into PyG Data
        x = torch.stack(self.node_features)
        edge_index = torch.tensor(
            self.edge_index, 
            dtype=torch.long
        )

        return Data(x=x, edge_index=edge_index)

class GNNActor(nn.Module):
    """
    Graph Attention Network actor producing task selection logits and suppressant predictions.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, init_weights=None):
        """
        Initialize GAT layers and output heads for task and suppressant prediction.

        Args:
            input_dim (int): input feature dimension.
            hidden_dim (int): hidden layer dimension.
            output_dim (int): output dimension for task selection logits.
            init_weights (str or None): path to pre-trained weights for initialization.
        """
        super().__init__()
        self.gat1 = GATConv(
            input_dim, 
            hidden_dim, 
            heads=2, 
            concat=True
        )
        self.gat2 = GATConv(
            hidden_dim * 2, 
            hidden_dim, 
            heads=2, 
            concat=True
        )
        self.final_gat = GATConv(
            hidden_dim * 2, 
            hidden_dim, 
            heads=1, 
            concat=False
        )

        # Task selection head (scalar logit per node)
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        # Suppressant prediction head (multi-class classifier)
        self.suppressant_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 classes: 0, 1, 2
        )

        if init_weights:
            self.load_weights(init_weights)

    def forward(self, x, edge_index, otheragent_indices=None):
        """
        Forward pass through gated attention layers.

        Args:
            x (Tensor): node features.
            edge_index (Tensor): graph connectivity.
            otheragent_indices (Tensor or None): indices for other-agent nodes.
        Returns:
            task_logits (Tensor): raw logits for task/hyperedge nodes.
            suppressant_logits (Tensor): prediction logits for other-agent suppressant classes.
        """
        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        h = F.relu(self.final_gat(h, edge_index))

        task_logits = self.out_layer(h).squeeze(-1)  # shape: [num_nodes]

        # Optional: compute suppressant prediction if requested
        otheragent_embeddings = h[otheragent_indices]                       # shape: [num_agents, hidden]
        suppressant_logits = self.suppressant_head(otheragent_embeddings)   # shape: [num_agents, 3]

        return task_logits, suppressant_logits
    
    
    def load_weights(self, weight_init):
        """
        Initialize weights of the GNN layers.

        Args:
            weight_init (str): path to pre-trained weights.
        """
        self.load_state_dict(weight_init.state_dict())

    def compute_loss(
            self, 
            data, 
            hyperedge_mask, 
            actions, 
            rewards, 
            otheragent_indices=None, 
            suppressant_labels=None, 
            intent_hyperedge_mask=None, 
            intent_labels=None,
            fire_outcomes=None, 
            λ_suppress=0.1, 
            λ_intent=0.2, 
            λ_belief=0.05,
            logging=False
        ):
        """
        Compute combined loss for task selection, suppressant prediction, intent, and belief updates.
        Logs individual components and total loss to CSV.

        Args:
            data (Data): graph input.
            hyperedge_mask (Tensor): mask for hyperedge nodes.
            actions (List[int]): list of selected actions for each agent.
            rewards (List[float]): list of rewards for each agent.
            otheragent_indices (Tensor or None): indices for other-agent nodes.
            suppressant_labels (List[int] or None): true labels for suppressant classes.
            intent_hyperedge_mask (Tensor or None): mask for intent hyperedges.
            intent_labels (List[int] or None): true labels for intent hyperedges.
            fire_outcomes (List[float] or None): fire outcomes for belief updates.
            λ_suppress, λ_intent, λ_belief (float): loss weighting factors.
        Returns:
            total_loss (Tensor): combined loss for the batch.
        """
        loss = []
        suppress_loss = []
        intent_loss = []
        belief_loss = []

        for i in range(len(actions)):
            logits, suppress = self.forward(data[i].x, data[i].edge_index, otheragent_indices[i] if otheragent_indices is not None else None)  # shape: [num_nodes]

            task_logits = logits[hyperedge_mask[i]]

            if task_logits.size(0) == 0:
                print(f"Warning: No valid logits for agent {i}. Skipping this agent.")
                continue

            action_tensor = torch.tensor(actions[i], dtype=torch.long)
            reward_tensor = torch.as_tensor(rewards[i], dtype=torch.float32)

            # === Task selection loss ===
            dist = Categorical(logits=task_logits)
            log_prob = dist.log_prob(action_tensor)
            task_loss = -log_prob * reward_tensor
            loss.append(task_loss)

            # === Suppressant prediction loss ===
            if suppressant_labels is not None:
                true_supp = torch.tensor([t.item() for t in suppressant_labels[i]], dtype=torch.long)  # shape: [num_other_agents]
                if suppress.size(0) > 0:
                    suppress_loss_i = F.cross_entropy(suppress, true_supp)
                    suppress_loss.append(suppress_loss_i)

                intent_mask = intent_hyperedge_mask[i]          # indices of intent hyperedges
                true_intent = intent_labels[i]                  # index of correct intent hyperedge

                for j in range(len(true_intent)):
                    start = j * len(intent_mask)/2
                    end = start + len(intent_mask)/2

                    intent_mask_j = intent_mask[int(start):int(end)]  # [task_size]
                    flat_index = true_intent[j] + int(start)    # shift by agent offset

                    if flat_index >= len(intent_mask):
                        continue  # safeguard

                    label_node_id = intent_mask[flat_index]
                    label_idx = intent_mask_j.tolist().index(label_node_id)  # get index within local slice

                    intent_logits_j = logits[intent_mask_j]  # shape: [task_size]
                    label_tensor = torch.tensor(label_idx, dtype=torch.long)

                    loss_j = F.cross_entropy(intent_logits_j.unsqueeze(0), label_tensor.unsqueeze(0))
                    intent_loss.append(loss_j)

            # === Belief update loss ===
            if fire_outcomes is not None:
                fire_outcomes_i = torch.tensor(fire_outcomes[i], dtype=torch.float32)  # shape: [num_tasks]
                if intent_logits_j.size(0) > 0 and suppress.size(0) > 0:
                    intent_probs = F.softmax(intent_logits_j, dim=0).unsqueeze(0)  # [1, num_tasks]
                    suppressant_probs = F.softmax(suppress, dim=1)  # [num_agents, 3]
                    present_probs = suppressant_probs[:, 1:].sum(dim=1)  # [num_agents]

                    belief_matrix = present_probs.unsqueeze(1) * intent_probs  # [num_agents, num_tasks]
                    task_beliefs = belief_matrix.sum(dim=0)  # [num_tasks]
                    fire_mask = (fire_outcomes_i == 0).float()
                    masked_beliefs = task_beliefs[:-1] * fire_mask
                    belief_loss_i = (masked_beliefs ** 2).mean()
                    belief_loss.append(belief_loss_i)

        task_loss = torch.stack(loss).mean() if loss else torch.tensor(0.0, requires_grad=True)
        suppressant_loss = torch.stack(suppress_loss).mean() if suppress_loss else torch.tensor(0.0, requires_grad=True)
        intent_loss = torch.stack(intent_loss).mean() if intent_loss else torch.tensor(0.0, requires_grad=True)
        belief_loss = torch.stack(belief_loss).mean() if belief_loss else torch.tensor(0.0, requires_grad=True)

        total_loss = (
            task_loss
            + λ_suppress * suppressant_loss
            + λ_intent * intent_loss
            + λ_belief * belief_loss
        )
        if logging:
            # Log losses to CSV
            header = ["task_loss", "suppressant_loss", "intent_loss", "belief_loss", "total_loss"]
            row = [task_loss.item(), suppressant_loss.item(), intent_loss.item(), belief_loss.item(), total_loss.item()]
            # write header if file not exists
            if not os.path.exists(LOSS_CSV_PATH):
                with open(LOSS_CSV_PATH, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
            # append row
            with open(LOSS_CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        if otheragent_indices is not None:
            print(f"Task Loss: {task_loss.item()}, Suppressant Loss: {suppressant_loss.item()}, Intent Loss: {intent_loss.item()}")
        else:
            print(f"Task Loss: {task_loss.item()}")

        return total_loss
        
    def forward_pass(self, data, hyper_mask, t_mappings, otheragent_indices=None):
        """
        Run inference: sample an action from task logits and determine intent mapping.
        Returns environment-formatted actions, raw logits, and intent index.

        Args:
            data (Data): graph input.
            hyper_mask (Tensor): mask for hyperedge nodes.
            t_mappings (Tensor): mapping of valid tasks to indices.
            otheragent_indices (Tensor or None): indices for other-agent nodes.
        Returns:
            env_actions (Tensor): environment-formatted actions.
            agent_actions (int): sampled action index.
            logits (Tensor): raw logits for task/hyperedge nodes.
            intent (int): index of the selected intent hyperedge.
        """
        logits, _ = self.forward(data.x, data.edge_index, otheragent_indices)
        logits = logits[hyper_mask]

        dist = Categorical(logits=logits)
        action = dist.sample()

        valid_tasks = t_mappings.tolist()

        if (action-1)==-1:
            intent=0
        else:
            intent=valid_tasks[action-1]

        env_actions = torch.tensor([[action-1, 0]], dtype=torch.int32)
        if (action-1)==-1:
            env_actions = torch.tensor([[-1, -1]], dtype=torch.int32)
        
        agent_actions = action.item()

        return env_actions, agent_actions, logits, intent

class GNNAgent(Agent):
    """
    Agent wrapper delegating observation processing and action sampling to GNNActor.

    Args:
        agent_name (str): name of the agent.
        parallel_envs (int): number of parallel environments.
        obs_dim (int): dimensionality of observation space.
        hidden_dim (int): dimensionality of hidden layers in GNNActor.
        init_weights (str or None): path to pre-trained weights for initialization.
    """
    def __init__(
        self,
        agent_name: str,
        parallel_envs: int,
        obs_dim=4,
        hidden_dim=32,
        init_weights=None,
    ):
        super().__init__(agent_name, parallel_envs)

        self.actions = torch.zeros((self.parallel_envs, 2), dtype=torch.int32)

        self.actor = GNNActor(input_dim=obs_dim, hidden_dim=hidden_dim, init_weights=init_weights)

        self.graph = Incidence_graph(node_size=obs_dim)

    def observe(self, observation: Tuple[Dict[str,Any], Any]) -> None:
        """
        Process batched observations and compute environment actions.
        Populates self.actions for subsequent `act` call.

        Args:
            observation (Tuple[Dict[str, Any], Any]): batched observations from the environment.
        """
        for batch in range(self.parallel_envs):
            obs_dict, t_map = observation
            data, hmask, omask, ohmask  = self.graph.build(
                obs_dict['self'][batch],
                obs_dict['others'][batch],
                obs_dict['tasks'].to_padded_tensor(-100)[batch],
                t_map['agent_action_mapping'][batch]
            )
            env_actions, _, _, _ = self.actor.forward_pass(
                data,
                hmask,
                t_map['agent_action_mapping'][batch],
                omask
            )

            self.actions[batch, :] = env_actions

    def act(self, action_space: free_range_rust.Space) -> torch.Tensor:
        """
        Return precomputed action tensor for the current timestep.

        Args:
            action_space (free_range_rust.Space): action space of the environment.
        Returns:
            torch.Tensor: action tensor for the current timestep.
        """
        return self.actions
        
class COMACritic(nn.Module):
    """
    Centralized critic using GAT to estimate joint Q-values for multi-agent actions.

    Args:
        input_dim (int): input feature dimension.
        hidden_dim (int): hidden layer dimension.
        action_dim (int): dimensionality of joint action space.
        num_agents (int): number of agents in the environment.
        output_dim (int): output dimension for Q-value prediction.
    """
    def __init__(self, input_dim, hidden_dim, action_dim, num_agents, output_dim=1):
        super(COMACritic, self).__init__()
        self.gat1 = GATConv(
            input_dim, 
            hidden_dim, 
            heads=2, 
            concat=True
        )
        self.gat2 = GATConv(
            hidden_dim * 2, 
            hidden_dim, 
            heads=2, 
            concat=True
        )
        self.gat3 = GATConv(
            hidden_dim * 2, 
            hidden_dim, 
            heads=1, 
            concat=False
        )
        self.q_out = nn.Linear(hidden_dim + num_agents * action_dim, output_dim)

    def forward(self, data, joint_action):
        """
        Forward pass computing pooled graph embedding and Q-value for joint actions.

        Args:
            data (Data): graph input.
            joint_action (Tensor): one-hot tensor of shape [batch_size, num_agents * action_dim].
        Returns:
            q (Tensor): predicted Q-value for the joint action.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))
        x = global_mean_pool(x, batch=None)

        if joint_action.dim() == 1:
            joint_action = joint_action.unsqueeze(0)  # ensure 2D

        elif joint_action.dim() == 3:
            joint_action = joint_action.squeeze(0)

        elif joint_action.dim() > 2:
            raise ValueError("joint_action should be 1D or 2D tensor")

        joint_input = torch.cat([x, joint_action], dim=-1)

        q = self.q_out(joint_input)

        return q

    def compute_advantage(self, agent_index, data, joint_action, logits):
        """
        Compute the advantage function for a given agent's action.

        Args:
            agent_index (int): index of the agent
            data (Data): graph input
            joint_action (Tensor): one-hot tensor of shape [batch_size, num_agents * action_dim]
            logits (Tensor): unnormalized logits for this agent of shape [batch_size, action_dim]
        Returns:
            advantage (Tensor): advantage value for the agent's action 
        """
        q_value = self.forward(data, joint_action).squeeze()

        baseline = 0.0
        num_actions = logits.shape[-1]
        probs = torch.softmax(logits, dim=-1)  # convert logits to probs

        for a_i_prime in range(num_actions):
            joint_action_prime = joint_action.clone()

            # Replace this agent's part of the joint action with a_i_prime (one-hot)
            start = agent_index * num_actions
            end = (agent_index + 1) * num_actions

            one_hot = torch.zeros_like(joint_action_prime[:, start:end])
            one_hot[:, a_i_prime] = 1.0
            joint_action_prime[:, start:end] = one_hot

            q_val = self.forward(data, joint_action_prime).squeeze()
            baseline += probs[a_i_prime] * q_val  # shape: [batch_size]

        advantage = q_value - baseline.detach()

        return advantage
    
    
    def compute_loss(self, data_list, joint_actions, target_q_values):
        """
        Args:
            data_list (List[Data]): list of torch_geometric.data.Data objects
            joint_actions (Tensor): tensor of shape [batch_size, num_agents * action_dim]
            target_q_values (Tensor): tensor of shape [batch_size]
        Returns:
            torch.Tensor: scalar MSE loss
        """
        q_preds = []
        for i, data in enumerate(data_list):
            q_pred = self.forward(data, joint_actions[i].unsqueeze(0))  # [1, 1]
            q_preds.append(q_pred.squeeze(0))  # [1]

        q_preds = torch.stack(q_preds)  # [batch_size]
        loss = F.mse_loss(q_preds, torch.tensor(np.array(target_q_values), dtype=torch.float32))
        return loss
