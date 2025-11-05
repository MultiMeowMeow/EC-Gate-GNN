import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention
import torch_geometric.nn as pyg_nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F
from torch_geometric.utils import degree
from de_calc_utils import dirichlet_normalized

class GNNStacked(nn.Module):
    """
    GNNStacked: Stacked GNN architecture with optional gating and post-FFN.
    Pipeline: x → GCNConv → BN → Activation → Dropout → Residual → (post_mlp) → x
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.model.num_layers
        self.residual = config.model.residual
        self.w_ecg = config.model.ec.w_ecg
        hidden_dim, output_dim = \
            config.model.hidden_dim, config.model.output_dim
        self.num_layers_w_computed_ec = config.model.ec.layer
        self.hidden_dim_gate = config.model.ec.hidden_dim_gate
        self.r = config.model.ec.repeat_factor

        self.node_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList([
            eval(config.model.conv)(hidden_dim) for _ in range(self.num_layers)
        ])
        if config.model.norm_type == 'ln':
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)
            ])
        else:
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(self.num_layers)
            ])
        self.norm_pos = config.model.norm_pos
        self.p = config.model.dropout

        # self.w_layer_ffn = config.model.w_layer_ffn
        # if self.w_layer_ffn:
        #     self.layer_ffns = nn.ModuleList([
        #         nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_layers)
        #     ])
        #     self.layer_ffn_bns = nn.ModuleList([
        #         nn.BatchNorm1d(hidden_dim) for _ in range(self.num_layers)
        # ])
        if self.w_ecg:
            self.fgs = nn.ModuleList([
                LogGate(hidden_dim // self.r, self.hidden_dim_gate) for _ in range(self.num_layers - 1)
            ])

        self.pool = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
            "attn": GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1)
            ))
        }.get(config.model.graph_pooling)

        self.post_proc = MLP(
            hidden_dim, hidden_dim, output_dim,
            num_layers=config.model.num_layers_postproc,
            act=nn.GELU,
            dropout=config.model.dropout,
            batch_norm=True
        )
        self.act = nn.GELU()
        self.device = 'cuda'

    def forward(self, G, measure_de=False):
        G = G.to(self.device)
        x = G.x.unsqueeze(-1) if G.x.dim() == 1 else G.x
        edge_attr = G.edge_attr.unsqueeze(-1) if G.edge_attr.dim() == 1 else G.edge_attr

        # stack of GNN layers: encoded x → agg → bn → gelu → dp → residual → post_mlp
        x = self.node_encoder(x)
        ec_acc = 1
        des = []
        for layer in range(self.num_layers):
            kwargs = {'edge_attr': edge_attr, 'ec_g': None}
            if self.w_ecg and layer > 0:
                idx = layer - 1
                if idx < self.num_layers_w_computed_ec:
                    cur_ec = G.exp_ea_ls[idx]
                    ec_acc += cur_ec
                else:
                    cur_ec = ec_acc / (self.num_layers_w_computed_ec +1)
                ec_g = self.fgs[idx](cur_ec)
                ec_g = ec_g.repeat_interleave(self.r, dim=-1)
                kwargs['ec_g'] = ec_g

            res = x
            x = self.convs[layer](x, G.edge_index, **kwargs)
            if self.norm_pos == "mid":
                x = self.norms[layer](x)
            x = self.act(x)
            x = F.dropout(x, self.p, training=self.training)
            if self.residual:
                x = x + res
            if self.norm_pos == "post":
                x = self.norms[layer](x)
            # if self.w_layer_ffn:
            #     res_ = x
            #     x = self.layer_ffns[layer](x)
            #     x = self.layer_ffn_bns[layer](x)
            #     x = self.act(x)
            #     x = F.dropout(x, self.p, training=self.training)
            #     x = x + res_

        if len(des):
            return torch.tensor(des)
        else:
            output = self.post_proc(self.pool(x, G.batch))
            return output



class GCNConv(pyg_nn.MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.act = nn.GELU()
        self.edge_encoder = BondEncoder(emb_dim)
        self.f_r = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr, ec_g=None):
        x = self.linear(x)
        edge_emb = self.edge_encoder(edge_attr)
        # <editor-fold desc="Norm Computation">
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # </editor-fold>
        agg = self.propagate(
            edge_index, x=x, edge_emb=edge_emb, norm=norm, ec_g=ec_g)
        self_upd = self.act(x + self.root_emb.weight) * 1./deg.view(-1, 1)
        return agg + self_upd

    def message(self, x_j, edge_emb, norm, ec_g):
        msg = self.act(x_j + edge_emb)
        msg = self.f_r(msg)
        if ec_g is not None:
            msg = (msg) * ec_g
        return norm.view(-1, 1) * msg

    def update(self, aggr_out):
        return aggr_out

class GINConv(pyg_nn.MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        hidden = emb_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2* hidden),
            torch.nn.LayerNorm(2* hidden), # BatchNorm1d
            torch.nn.GELU(),
            torch.nn.Linear(2* hidden, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.f_r = torch.nn.Linear(emb_dim, emb_dim) # To Dense/un-activated rep.
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr, ec_g=None):
        edge_emb = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x
                       + self.propagate(edge_index, x=x, edge_emb=edge_emb, ec_g=ec_g))
        return out

    def message(self, x_j, edge_emb, ec_g):
        msg = self.act(x_j + edge_emb)
        msg = self.f_r(msg)
        if ec_g is not None:
            msg = msg * ec_g
        return msg

    def update(self, aggr_out):
        return aggr_out


class GatedConv(pyg_nn.MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add')
        self.f_k = nn.Linear(emb_dim, emb_dim)
        self.f_q = nn.Linear(emb_dim, emb_dim)
        self.f_v = nn.Linear(emb_dim, emb_dim)
        self.f_skip = nn.Linear(emb_dim, emb_dim, bias=False)
        self.edge_encoder = BondEncoder(emb_dim)

        self.f_r = nn.Linear(emb_dim, emb_dim, bias=False)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr, g=None):
        out = self.propagate(edge_index, k=x, q=x, v=x, edge_attr=edge_attr, g=g)
        out = out + self.f_skip(x) # residual connection.
        return out

    def message(self, k_i, q_j, v_j, edge_attr, g):
        k_i = self.f_k(k_i)
        q_j = self.f_q(q_j) # f_q
        v_j = self.f_v(v_j) # f_v
        e = k_i + q_j + self.edge_encoder(edge_attr) # shape [num_edges, emb_dim]

        msg = torch.sigmoid(e) * v_j
        if g is not None:
            msg = self.f_r(msg) * g
        return msg


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act, dropout=0.0, batch_norm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class LogGate(nn.Module):
    def __init__(self, out_dim, hidden):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim)
        )
        last: nn.Linear = self.f[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, ec):
        x = torch.log1p(ec).unsqueeze(-1)
        x = self.f(x)
        x = torch.sigmoid(x)
        return x