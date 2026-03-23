import torch
from torch import nn


# Layer 7: Physical Consistency Layer (space-time smoothing)
class PhysicalConsistencyLayer(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float, sigma: float):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, state: torch.Tensor, latlon: torch.Tensor, valid_mask: torch.Tensor):
        # state: (B, T, 1)
        state = state * valid_mask.unsqueeze(-1).float()
        prev = torch.cat([state[:, :1], state[:, :-1]], dim=1)

        if self.gamma == 0 or latlon is None:
            neighbor = torch.zeros_like(state)
        else:
            # Spatial smoothing within batch using Gaussian kernel
            dist = torch.cdist(latlon, latlon)  # (B, B)
            weights = torch.exp(- (dist ** 2) / (2 * self.sigma ** 2))
            eye = torch.eye(weights.size(0), device=weights.device)
            weights = weights * (1.0 - eye)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

            s = state.squeeze(-1)  # (B, T)
            neighbor = torch.matmul(weights, s).unsqueeze(-1)  # (B, T, 1)

        consistent = (self.alpha * state) + (self.beta * prev) + (self.gamma * neighbor)
        return consistent


class STWQHRNet(nn.Module):
    def __init__(self, config: dict, meta: dict):
        super().__init__()

        model_cfg = config["model"]

        num_district = len(meta["vocab"]["district"])
        num_block = len(meta["vocab"]["block"])
        num_village = len(meta["vocab"]["village"])
        num_season = len(meta["vocab"]["season"])
        num_source = len(meta["vocab"]["source_type"])
        num_numeric = len(meta["numeric_cols"])

        self.emerging_above_idx = meta["emerging_above_idx"]

        # Layer 2: Embedding Layer (district/block/village/season/source)

        self.district_emb = nn.Embedding(num_district, model_cfg["district_emb_dim"])
        self.block_emb = nn.Embedding(num_block, model_cfg["block_emb_dim"])
        self.village_emb = nn.Embedding(num_village, model_cfg["village_emb_dim"])
        self.season_emb = nn.Embedding(num_season, model_cfg["season_emb_dim"])
        self.source_emb = nn.Embedding(num_source, model_cfg["source_emb_dim"])

        loc_emb_dim = model_cfg["district_emb_dim"] + model_cfg["block_emb_dim"] + model_cfg["village_emb_dim"]

        # Layer 3: Spatial Encoding Layer
        self.spatial_mlp = nn.Sequential(
            nn.Linear(loc_emb_dim + 2, model_cfg["spatial_hidden"]),
            nn.ReLU(),
            nn.Linear(model_cfg["spatial_hidden"], model_cfg["spatial_dim"]),
        )

        # District-level aggregate features (fed into the core sequence)
        self.district_mlp = nn.Sequential(
            nn.Linear(num_numeric, model_cfg["district_hidden"]),
            nn.ReLU(),
            nn.Linear(model_cfg["district_hidden"], model_cfg["district_dim"]),
        )

        input_dim = num_numeric + model_cfg["season_emb_dim"] + model_cfg["source_emb_dim"] + model_cfg["spatial_dim"] + model_cfg["district_dim"]

        # Layer 4: Temporal Encoding Layer (projection + positional embedding)
        self.input_proj = nn.Linear(input_dim, model_cfg["d_model"])
        self.pos_emb = nn.Embedding((config["data"]["future_end_year"] - config["data"]["start_year"] + 1) * 3, model_cfg["d_model"])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg["d_model"],
            nhead=model_cfg["n_heads"],
            dim_feedforward=model_cfg["ff_dim"],
            dropout=model_cfg["dropout"],
            batch_first=True,
        )
        # Layer 5: Spatio-Temporal Transformer Encoder (core)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg["n_layers"])


        # Layer 6: Contamination State Layer
        self.state_head = nn.Linear(model_cfg["d_model"], 1)


        # Layer 7: Physical Consistency Layer
        self.physical = PhysicalConsistencyLayer(
            alpha=model_cfg["phys_alpha"],
            beta=model_cfg["phys_beta"],
            gamma=model_cfg["phys_gamma"],
            sigma=model_cfg["phys_sigma"],
        )


        # Layer 8: Risk Scoring Layer
        self.risk_head = nn.Linear(1, 1)


        # Layer 9: WQI Prediction Head
        self.wqi_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, len(meta["vocab"]["wqi_class"])),
        )


        # Layer 10: HHI Prediction Head
        self.hhi_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, len(meta["vocab"]["hhi_level"])),
        )

    def forward(self, x_num, x_season, x_source, loc_cat, latlon, district_feat, valid_mask, return_intermediates: bool = False):
        # Layer 1: Input Layer (raw structured tensors are passed in)
        # Embeddings
        district = self.district_emb(loc_cat[:, 0])
        block = self.block_emb(loc_cat[:, 1])
        village = self.village_emb(loc_cat[:, 2])
        loc_emb = torch.cat([district, block, village], dim=-1)

        # Spatial encoding
        spatial_in = torch.cat([loc_emb, latlon], dim=-1)
        spatial_vec = self.spatial_mlp(spatial_in)  # (B, spatial_dim)

        district_vec = self.district_mlp(district_feat)  # (B, district_dim)

        # Temporal features
        season_emb = self.season_emb(x_season)
        source_emb = self.source_emb(x_source)

        spatial_expand = spatial_vec.unsqueeze(1).expand(-1, x_num.size(1), -1)
        district_expand = district_vec.unsqueeze(1).expand(-1, x_num.size(1), -1)
        x = torch.cat([x_num, season_emb, source_emb, spatial_expand, district_expand], dim=-1)
        x = self.input_proj(x)

        # Positional encoding
        positions = torch.arange(x.size(1), device=x.device)
        pos = self.pos_emb(positions)
        x = x + pos.unsqueeze(0)

        x_embed = x

        # Safe causal attention by packing valid timesteps (removes leading padding).
        pad_mask = ~valid_mask
        lengths = valid_mask.sum(dim=1)
        max_len = int(lengths.max().item())

        batch_size, seq_len, dim = x.size()
        x_packed = x.new_zeros((batch_size, max_len, dim))
        pad_trim = torch.ones((batch_size, max_len), dtype=torch.bool, device=x.device)
        idx_list = []
        for b in range(batch_size):
            idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(1)
            idx_list.append(idx)
            if idx.numel() > 0:
                x_packed[b, : idx.numel()] = x[b, idx]
                pad_trim[b, : idx.numel()] = False

        causal_mask = torch.triu(torch.ones(max_len, max_len, device=x.device), diagonal=1).bool()
        x_packed = self.transformer(x_packed, mask=causal_mask, src_key_padding_mask=pad_trim)
        x_packed = x_packed.masked_fill(pad_trim.unsqueeze(-1), 0.0)

        # Scatter back to original timeline positions
        x_full = x.new_zeros((batch_size, seq_len, dim))
        for b, idx in enumerate(idx_list):
            if idx.numel() > 0:
                x_full[b, idx] = x_packed[b, : idx.numel()]
        x = x_full

        # Contamination state
        state = self.state_head(x)

        # Physical consistency
        state_consistent = self.physical(state, latlon, valid_mask)

        # Risk scoring
        risk = torch.sigmoid(self.risk_head(state_consistent))

        # Heads
        wqi_logits = self.wqi_head(risk)

        above_em = x_num[:, :, self.emerging_above_idx]
        hhi_in = torch.cat([risk, above_em], dim=-1)
        hhi_logits = self.hhi_head(hhi_in)

        intermediates = {
            "loc_emb": loc_emb,
            "spatial_vec": spatial_vec,
            "district_vec": district_vec,
            "x_embed": x_embed,
            "transformer_out": x,
            "state": state,
            "state_consistent": state_consistent,
            "risk": risk,
            "wqi_logits": wqi_logits,
            "hhi_logits": hhi_logits,
        }

        if return_intermediates:
            return wqi_logits, hhi_logits, risk, state_consistent, intermediates

        return wqi_logits, hhi_logits, risk, state_consistent
