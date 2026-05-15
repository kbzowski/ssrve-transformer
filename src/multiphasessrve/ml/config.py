"""Configuration for the SSRVE Transformer (DETR-style) training pipeline.

Per-point schema: (x, y, phase_id, nuc_ratio), 4 features.

Architecture: pure transformer set prediction (DETR-like).
- CoolingCurveEncoder (TransformerEncoder) processes the cooling curve.
- PointDecoder (TransformerDecoder with learned queries) emits nucleation slots.
- Noise z is injected as a single memory token. Not a VAE: no KL term, no
  posterior encoder; the noise token is included for optional stochasticity
  without the teacher-forcing leakage of a CVAE.

Cooling curve: densely sampled (time, temperature) pairs along the linear
T(t) between t_start and t_ms, padded to curve_max_len.

Number of nucleation points varies per CR (19 at CR=0.5 up to 70 at CR>=70).
All samples are padded to num_points with active_mask.
"""

from dataclasses import dataclass


@dataclass
class CVAEConfig:
    # Architecture
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 3
    latent_dim: int = 32
    dropout: float = 0.1
    dim_feedforward: int = 512

    # Per-point schema
    num_points: int = 70  # max, padded with active_mask
    params_per_point: int = 4  # x, y, phase_id, nuc_ratio
    num_phases: int = 4
    max_nucleation_ratio: float = 0.3

    # Inference-only constants (functionally inert in SSRVE growth)
    constant_compatibility: float = 0.5
    constant_threshold: float = 0.5

    # Cooling curve (variable-length sequence handled by transformer padding+mask)
    curve_input_dim: int = 2  # (time, temperature) per point
    curve_max_len: int = 32  # padding limit
    curve_num_samples: int = 16  # dense sampling along T(t)
    t_start: float = 830.0  # deg C, austenite start
    t_ms: float = 400.0  # deg C, martensite start for S355J2
    max_time: float = 1300.0  # seconds, max cooling time

    # Normalisation (filled by build_cache from train-split stats)
    cr_log_mean: float = 0.0
    cr_log_std: float = 1.0
    gs_mean: float = 0.0
    gs_std: float = 1.0
    max_grain_size: float = 15.0  # um, naive normalisation fallback

    # Sample filtering (in dataloader / cache builder)
    filter_max_frac_err: float = 0.03
    filter_max_gs_err_pct: float = 10.0
    # Optional CR range restriction during cache build; useful for low-CR
    # ferrite-dominated PoC experiments (e.g. cr_max=20).
    cr_min: float | None = None
    cr_max: float | None = None

    # Class balance
    sampler_weight_pow: float = 0.5  # weight = 1 / n_per_cr ** pow

    # Best-of selection per CR
    best_of_per_cr: int = 200

    # Reproducibility. None gives non-deterministic behaviour; ablation runs
    # require a fixed seed because per-variant differences are otherwise
    # dominated by seed variance.
    random_seed: int | None = 42

    # Train / val / test split
    split_seed: int = 42
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    holdout_cr_indices: tuple[int, ...] = (7, 9)  # OOD eval set

    # Augmentation
    enable_augmentation: bool = False

    # Sinkhorn loss (Option B).
    # Feature vector: [feat_w_nuc * nuc, phase_embed(softmax(logits))]
    # phase_embed is a learned Linear(num_phases, phase_embed_dim).
    # Blur is annealed linearly from sinkhorn_blur_start (smooth gradients,
    # global structure) over sinkhorn_blur_warmup_epochs (~30% of total)
    # down to sinkhorn_blur_end (sharp matching).
    phase_embed_dim: int = 4
    # If False, freeze phase_embed at orthogonal init (non-trainable).
    phase_embed_trainable: bool = True
    feat_w_nuc: float = 3.0  # rescales nuc from [0, 0.3] to [0, 0.9]
    sinkhorn_blur_start: float = 0.2
    sinkhorn_blur_end: float = 0.05
    sinkhorn_blur_warmup_epochs: int = 60
    sinkhorn_scaling: float = 0.9

    # Geometry / phases
    grid_size: int = 500
    real_size: float = 50.0  # um

    # Soft Voronoi (end-to-end xy training). Gradient flows from fraction
    # losses through SoftVoronoi and the distance field to the xy head.
    sv_grid_size: int = 64          # G x G pixel grid (more pixels = more accurate, more memory)
    sv_temp_start: float = 0.05    # initial temperature (smooth assignments)
    sv_temp_end: float = 0.005     # final temperature (sharp Voronoi)
    sv_temp_warmup_epochs: int = 150  # linear decay over this many epochs
    lambda_sv_frac: float = 5.0    # SoftVoronoi phase-fraction loss weight
    lambda_sv_gs: float = 3.0      # SoftVoronoi ferrite grain-size loss weight

    # Apollonius / Johnson-Mehl CA matching.
    # The JM-CA implementation in ssrve_algs.py uses periodic BC, so the
    # training-time loss must enforce the same boundary condition; otherwise
    # edge artefacts depress the per-grain coverage near the domain border.
    sv_periodic: bool = True
    # Use the additively weighted (Apollonius) diagram with nuc_ratio as the
    # weight. In the zero-temperature limit this matches JM-CA with constant
    # growth velocity. Switching this on changes the SoftVoronoi distance
    # from squared to plain Euclidean.
    sv_use_nuc_weights: bool = True
    # Scale that maps nuc_ratio in [0, 0.3] to a delay in distance units
    # [0, 1]. With the sqrt mapping (delay = scale * sqrt(nuc_ratio)) the
    # empirical optimum is s = 0.3.
    sv_nuc_weight_scale: float = 0.3
    # Apollonius delay functional form (A3 ablation): "sqrt" (baseline),
    # "linear", "const". "const" requires sv_nuc_ratio_const != None
    # (A3b uses w_bar measured on the L0 val set).
    sv_apollonius_mode: str = "sqrt"
    sv_nuc_ratio_const: float | None = None
    # A1 ablation: force z = zeros at training and inference, consistent
    # with the empirical fact that the model ignores z (std ~ 1e-5).
    disable_z_token: bool = False
    # Gumbel-Softmax straight-through: enabled in the late training phase
    # (see sv_gumbel_start_epoch) for hard, discrete assignments that still
    # carry gradients. Decoupled Gumbel-ST (arXiv:2410.13331).
    sv_gumbel_ste: bool = False
    sv_gumbel_start_epoch: int = 100
    # Uniform coverage: Sinkhorn distance between the predicted xy
    # distribution and a uniform reference grid. Prevents a degenerate
    # "stripe" collapse where all nuclei cluster in a horizontal band
    # (correct fractions, wrong spatial structure).
    lambda_coverage: float = 3.0
    coverage_ref_size: int = 16    # uniform reference grid = coverage_ref_size^2 points

    # CCT empirical fraction prior (Stage 3).
    # One-sided inverse-frequency penalty on the decoder marginal pred_marg.
    # Penalises pred_marg_k < target_frac_k per phase k, weighted by
    # 1/mean_target_frac_k so minority phases (bainite ~4%) get
    # roughly 25x more weight than ferrite (~70%). Acts on the decoder
    # marginal, not on fraction_head. Addresses decoder misallocation of
    # 1-2 minority nuclei at inference relative to training.
    lambda_cct: float = 0.0

    # Phase separation regulariser (Stage 2).
    # Hinge loss max(0, delta - pairwise_dist) per off-diagonal pair of
    # phase_embed.weight rows; prevents the embedding from collapsing to
    # near-equal vectors (we observed d(B, M) ~ 0.008 with the standard
    # Sinkhorn objective). Active when lambda_phase_sep > 0. Requires
    # phase_embed_trainable = True.
    lambda_phase_sep: float = 0.0
    phase_sep_delta: float = 0.5

    # Morphological losses (S2, boundary ratio, pair distribution).
    # Targets are computed once from dataset PKL files. Differentiable
    # versions are computed from the SoftVoronoi pixel_phase_2d map
    # (S2 and boundary ratio) and the predicted xy positions (pair
    # distribution). Each is phase-fraction-weighted so absent phases
    # contribute zero automatically.
    lambda_s2: float = 5.0           # S2 two-point statistics weight
    lambda_boundary: float = 2.0     # boundary ratio weight
    lambda_pair_dist: float = 1.0    # same-phase pair distribution weight

    # Training
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 500

    # Loss weights (Option B)
    lambda_sinkhorn: float = 1.0
    lambda_marg_kl: float = 5.0
    lambda_active: float = 2.0  # BCE on active mask
    lambda_n_active: float = 1.0  # auxiliary regression of n_active
    lambda_fractions: float = 2.0  # complements the marginal KL term
    lambda_grain_size: float = 5.0

    # Inference monitoring: every N epochs run a generation pass without
    # teacher forcing and log the resulting losses.
    inference_eval_every: int = 20

    early_stopping_patience: int = 100

    # Data paths
    dataset_dir: str = "./dataset"
    cache_dir: str = "./cache"

    # Device
    device: str = "cuda"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./runs"

    @property
    def total_params(self) -> int:
        return self.num_points * self.params_per_point

    def cooling_curve_from_cr(self, cr: float) -> list[tuple[float, float]]:
        """Return curve_num_samples evenly spaced (t, T) points along
        T(t) = t_start - cr * t between t_start and t_ms.

        A dense curve is needed for the transformer encoder to have
        anything meaningful to attend over; with two points (start/end)
        the encoder degenerates to an MLP on a single scalar (CR).
        """
        t_end = (self.t_start - self.t_ms) / cr
        curve = []
        for i in range(self.curve_num_samples):
            frac = i / (self.curve_num_samples - 1)
            t = frac * t_end
            temp = self.t_start - cr * t
            curve.append((t, temp))
        return curve

    def normalize_curve(
        self, curve: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Normalise cooling curve: time by max_time, temperature by t_start."""
        return [(t / self.max_time, temp / self.t_start) for t, temp in curve]
