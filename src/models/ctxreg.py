"""Context-conditioned regressor (model.arch=ctxreg).

A no-context sensor-set regressor whose sensor tokens also receive a deployment
embedding computed from the context:

    context point (records + label)
        -> sensor-set encoder with the label on every sensor token
        -> one vector per context point
        -> small transformer over the context points, with a readout token
        -> deployment embedding e
    target point: sensor tokens + gate(e) -> sensor-set encoder -> head

The gate is a linear map initialized to zero, so at initialization the model is
the no-context regressor, and the context enters before the target's records
are compressed (the PFN consults it after). With spatial.ctx_gate=false the
context branch is not built and the model is the plain regressor.

spatial.ctx_mode selects how the context reaches the target:
    summary  the deployment embedding above: one vector for every target of the
             deployment (a conditional neural process).
    attn     target-specific attention, as in a PFN or a transformer neural
             process: before every layer of the target's sensor-set encoder,
             the target's readout token attends to the individual context
             tokens (the context transformer's outputs, the summary among
             them). Targets attend to the context only, never to each other.
             Each attention's output map is initialized to zero.
    both     summary and attention.

Lighter variants (defaults keep the full model, 12.95 M parameters in attn mode):
    spatial.ctx_share=true      the context points run through the target
                                encoder's transformer layers (one sensor encoder
                                for both, as the PFN) and keep their own input
                                map, which receives the label, readout token
                                and norm. Saves 4.74 M at width 256 x 6.
    spatial.ctx_pool_layers=0   no transformer over the context points: the
                                targets attend to the per-point encodings
                                directly (attn mode only). Saves 1.58 M.
    spatial.ctx_attn_layers     the target-encoder layers, counted from 1, that
                                are preceded by a context attention (default:
                                all). 0.26 M each.

Same interface as the models in anp.py and pfn.py (forward signature and the
5-tuple (mean, var, loss, kl, nll)). The front end is always the sensor-set
one: model.spatial.encoder must be `set`, and the width is spatial.set_d_model
(num_hidden is not used).

Precision: spatial.set_bf16 runs the encoders in bfloat16 while training only.
The output head is always float32 and evaluation is all float32: bfloat16 has
8 significant bits, which rounds a coordinate of 200 to 400 m to steps of
1.5 m (0.1 m of mean error on B2).
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .anp import gaussian_nll, reduce_nll, DEFAULT_VAR_FLOOR


def _encoder(d, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ContextAttention(nn.Module):
    """Pre-norm cross-attention from point tokens (B, m, d) to a deployment's
    context tokens (B, K, d). Returns the residual update; the output map is
    initialized to zero."""

    def __init__(self, d, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, q, keys, pad=None):               # pad (B, K): True = not a key
        return self.attn(self.norm(q), keys, keys, key_padding_mask=pad, need_weights=False)[0]


class SetBody(nn.Module):
    """Transformer over one point's sensor tokens, read out through a learned
    token (as anp.SensorSetEncoder). Optional per-point inputs: extra (P, E),
    appended to every sensor token's features; cond (P, d), added to every token
    after the input map; cross = (blocks, keys (B, K, d), pad (B, K)), where the
    P points are B groups of P // B and, before each layer, every point's readout
    token attends to its group's keys through that layer's block (blocks maps
    the layer index, as a string, to its block; layers without one are plain).
    With layers=0 the body has no transformer layers of its own and forward is
    given another body's through `layers` (a shared encoder)."""

    def __init__(self, tau, d, layers, heads, dropout=0.0, extra=0):
        super().__init__()
        self.norm_mode = "standardize"                  # the trainer sets the stats
        self.register_buffer("x_mean", t.zeros(1))
        self.register_buffer("x_scale", t.ones(1))
        self.inp = nn.Sequential(nn.Linear(tau + 2 + extra, d), nn.GELU(), nn.Linear(d, d))
        self.readout = nn.Parameter(t.zeros(1, 1, d))
        self.layers = _encoder(d, heads, layers, dropout) if layers > 0 else None
        self.norm = nn.LayerNorm(d)

    def set_acoustic_stats(self, mean, std):
        self.x_mean.fill_(float(mean))
        self.x_scale.fill_(float(std) + 1e-8)

    def forward(self, tok, rel, kpm=None, extra=None, cond=None, cross=None, layers=None):
        """tok (P, S, tau) standardized records, rel (P, S, 2) sensor positions,
        kpm (P, S) True = sensor absent. Returns (P, d)."""
        layers = self.layers if layers is None else layers
        P, S, _ = tok.shape
        feats = [tok, rel.to(tok.dtype)]
        if extra is not None:
            feats.append(extra[:, None, :].expand(P, S, extra.size(-1)).to(tok.dtype))
        h = self.inp(t.cat(feats, -1))
        h = t.cat([self.readout.expand(P, -1, -1).to(h.dtype), h], 1)
        if cond is not None:
            h = h + cond[:, None, :].to(h.dtype)
        if kpm is not None:
            kpm = t.cat([t.zeros(P, 1, dtype=t.bool, device=tok.device), kpm], 1)
        if cross is None:
            return self.norm(layers(h, src_key_padding_mask=kpm)[:, 0])
        blocks, keys, pad = cross
        B = keys.size(0)
        for i, layer in enumerate(layers.layers):
            if str(i) in blocks:
                q = h[:, 0].reshape(B, P // B, -1)
                q = q + blocks[str(i)](q, keys, pad).to(q.dtype)
                h = t.cat([q.reshape(P, 1, -1), h[:, 1:]], 1)
            h = layer(h, src_key_padding_mask=kpm)
        return self.norm(h[:, 0])


class ContextRegressor(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim, dropout=0.1, spatial_cfg=None):
        super().__init__()
        get = (spatial_cfg or {}).get
        if not get("enabled", False) or str(get("encoder", "legacy")).lower() != "set":
            raise ValueError("ctxreg needs model.spatial.enabled=true and spatial.encoder=set")
        S = int(get("n_sensors", 10))
        assert input_dim % S == 0, f"feat_dim {input_dim} not divisible by n_sensors {S}"
        self.n_sensors, self.tau, self.output_dim = S, input_dim // S, output_dim
        d = self.d_model = int(get("set_d_model", 256))
        layers, heads = int(get("set_layers", 6)), int(get("set_heads", 8))
        drop = float(get("set_dropout", 0.0))
        self.pos_scale = float(get("set_pos_scale", 100.0))
        self.bf16 = bool(get("set_bf16", True))
        self.ctx_gate = bool(get("ctx_gate", True))
        self.ctx_mode = str(get("ctx_mode", "summary")).lower()
        if self.ctx_mode not in ("summary", "attn", "both"):
            raise ValueError(f"spatial.ctx_mode must be summary|attn|both, got {self.ctx_mode}")
        # Training only: probability of hiding a sample's whole context, so the
        # model keeps working (as the plain regressor) when it has none.
        self.ctx_drop = float(get("ctx_drop", 0.0))
        # Lighter variants, see the module docstring.
        self.ctx_share = bool(get("ctx_share", False))
        pool_layers = int(get("ctx_pool_layers", 2))
        attn_layers = get("ctx_attn_layers", None)
        attn_layers = list(range(1, layers + 1)) if attn_layers is None else sorted(int(i) for i in attn_layers)
        if any(i < 1 or i > layers for i in attn_layers):
            raise ValueError(f"spatial.ctx_attn_layers must lie in 1..{layers}, got {attn_layers}")
        if pool_layers == 0 and self.ctx_gate and self.ctx_mode != "attn":
            raise ValueError("spatial.ctx_pool_layers=0 needs spatial.ctx_mode=attn "
                             "(the summary is the context transformer's readout)")

        # Named spatial_encoder so the trainer sets the record statistics on it.
        self.spatial_encoder = SetBody(self.tau, d, layers, heads, drop)
        if self.ctx_gate:
            self.ctx_encoder = SetBody(self.tau, d, 0 if self.ctx_share else int(get("ctx_layers", layers)),
                                       heads, drop, extra=output_dim)
            self.ctx_readout = nn.Parameter(t.zeros(1, 1, d))
            self.ctx_pool = _encoder(d, heads, pool_layers, drop) if pool_layers > 0 else None
            self.ctx_norm = nn.LayerNorm(d)
            if self.ctx_mode in ("summary", "both"):
                self.gate = nn.Linear(d, d)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)
            if self.ctx_mode in ("attn", "both"):
                # Keyed by the 0-based layer index (the keys of the earlier ModuleList).
                self.ctx_attn = nn.ModuleDict({str(i - 1): ContextAttention(d, heads) for i in attn_layers})
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * output_dim))
        # Variance floor (see anp.set_var_floor), not saved in the state_dict.
        self.register_buffer("var_floor", t.tensor(DEFAULT_VAR_FLOOR), persistent=False)

    def _tokens(self, x):
        """(..., tau * S) tau-major records -> standardized (..., S, tau)."""
        se = self.spatial_encoder
        tok = x.reshape(*x.shape[:-1], self.tau, self.n_sensors).transpose(-1, -2)
        return (tok - se.x_mean) / se.x_scale

    def embed_context(self, context_x, context_y, rel, kpm, context_mask=None):
        """Context tokens (B, 1 + n, d) and their padding mask (True = not a key)
        from the context points marked in context_mask. Token 0 is the deployment
        embedding (without a context transformer, a learned constant); it is
        always present, also with an empty context."""
        B, n, _ = context_x.shape
        dev = context_x.device
        km = (t.ones(B, n, dtype=t.bool, device=dev) if context_mask is None
              else context_mask.to(t.bool))
        if self.training and self.ctx_drop > 0:
            km = km & (t.rand(B, 1, device=dev) >= self.ctx_drop)
        c = t.zeros(B, 1 + n, self.d_model, device=dev, dtype=self.ctx_readout.dtype)
        c[:, 0] = self.ctx_readout[0, 0]
        if km.any():                                   # encode the valid points only
            b = km.nonzero(as_tuple=True)[0]
            enc = self.ctx_encoder(self._tokens(context_x[km]), rel[b],
                                   None if kpm is None else kpm[b], extra=context_y[km],
                                   layers=self.spatial_encoder.layers if self.ctx_share else None)
            c = c.to(enc.dtype)
            c[:, 1:][km] = enc
        pad = t.cat([t.zeros(B, 1, dtype=t.bool, device=dev), ~km], 1)
        if self.ctx_pool is not None:
            c = self.ctx_pool(c, src_key_padding_mask=pad)
        return self.ctx_norm(c), pad

    def forward(self, context_x, context_y, target_x, target_y=None,
                beta: float = 1.0, predict_with_prior: bool = False,
                sensor_pos=None, sensor_mask=None,
                context_mask=None, target_mask=None):
        # beta and predict_with_prior keep the interface of the latent models
        # and have no effect. sensor_mask: optional bool (B, n_sensors), True for
        # present sensors, shared by the context and the targets.
        if sensor_pos is None:
            raise ValueError("ctxreg needs sensor_pos")
        B, m, _ = target_x.shape
        S = self.n_sensors
        pos = sensor_pos[..., :2]
        rel = (pos - pos.mean(1, keepdim=True)) / self.pos_scale          # (B, S, 2)
        kpm = None if sensor_mask is None else ~sensor_mask.to(t.bool)    # True = absent
        bf16 = self.bf16 and self.training and target_x.is_cuda
        with t.autocast("cuda", dtype=t.bfloat16, enabled=bf16):
            cond = cross = None
            if self.ctx_gate:
                c, pad = self.embed_context(context_x, context_y, rel, kpm, context_mask)
                if self.ctx_mode in ("summary", "both"):
                    cond = self.gate(c[:, 0])[:, None, :].expand(B, m, -1).reshape(B * m, -1)
                if self.ctx_mode in ("attn", "both"):
                    cross = (self.ctx_attn, c, pad)
            h = self.spatial_encoder(
                self._tokens(target_x).reshape(B * m, S, self.tau),
                rel[:, None].expand(B, m, S, 2).reshape(B * m, S, 2),
                None if kpm is None else kpm[:, None].expand(B, m, S).reshape(B * m, S),
                cond=cond, cross=cross)
        with t.autocast("cuda", enabled=False):          # float32 head, see the module docstring
            out = self.head(h.float())
        mean, raw = out.view(B, m, -1).chunk(2, dim=-1)
        var = self.var_floor + F.softplus(raw)

        if target_y is not None:
            nll = reduce_nll(gaussian_nll(mean, var, None, target_y), target_mask)
            loss = nll
        else:
            nll = None
            loss = None
        return mean, var, loss, None, nll                    # kl always None
