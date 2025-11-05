import numpy as np
from torch_geometric.utils import to_dense_adj
import torch


def compute_ea(edge_index, num_nodes, L):
    num_edges = edge_index.size(1)
    edge_counts = []
    # Step 1: Build edge lookup table
    # Convert (src, tgt) to unique IDs: src * N + tgt
    edge_ids = edge_index[0] * num_nodes + edge_index[1]  # shape: (num_edges,)
    edge_id_map = torch.full((num_nodes * num_nodes,), -1, dtype=torch.long)
    edge_id_map[edge_ids] = torch.arange(num_edges)  # edge ID per (src, tgt)
    # Step 2: Convert edge_index to dense adj for walk computation (as numpy for boolean ops)
    A = to_dense_adj(edge_index).squeeze(0).numpy().astype(bool)  # (N, N)
    Ak_minus1 = A
    Ak_prev = np.eye(num_nodes, dtype=bool)
    for k in range(2, L + 1):
        # Step 3: Compute reachable node pairs at hop k (Ak)
        Ak = (A.astype(np.uint8) @ Ak_minus1.astype(np.uint8).T) > 0
        Ak_prev += Ak_minus1  # accumulate previous reachability
        Ak_filt = np.logical_and(Ak, Ak_prev == False)  # new pairs only

        temp = torch.zeros(num_edges, dtype=torch.long)
        if Ak_filt.any():
            # Step 4: Find contributing 1-hop edges
            r, c = np.nonzero(Ak_filt)  # new (tgt, src)
            r_index, src = np.nonzero(np.logical_and(A[r, :], (Ak_minus1.T)[c, :]))
            tgt = r[r_index]
            pairs = torch.tensor(np.stack((src, tgt)), dtype=torch.long)  # shape: (2, N)
            # Step 5: Vectorized edge ID lookup and count accumulation
            flat_ids = pairs[0] * num_nodes + pairs[1]  # encode (src, tgt)
            valid_mask = edge_id_map[flat_ids] >= 0
            edge_idx = edge_id_map[flat_ids[valid_mask]]  # valid edge IDs
            temp = temp.scatter_add(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.long))

        # Step 6: Save per-hop edge count
        edge_counts.append(temp.to(torch.uint8))
        Ak_minus1 = Ak
    return edge_counts

if __name__ == "__main__":
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 4, 3, 5, 6, 4, 6, 4, 5]])
    ls = compute_ea(edge_index, 7, L=6)
    print(ls)