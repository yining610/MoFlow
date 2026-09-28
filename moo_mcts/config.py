from dataclasses import dataclass, field


@dataclass
class SearchConfig:

    n_trials: int = 200
    max_depth: int = 8  # H: max atomic edits before forced terminate
    branching: int = 4  # b: candidate decisions the proposer returns per node
    beam_k: int = 3  # k: completion-beam width
    chance_samples: int = 3  # R: LLM realizations sampled per chance node
    selector: str = "czt"  # tree policy: "czt" | "pareto" | "hypervolume" | "chebyshev"
    czt_c: float = 4.0  # CZT confidence-radius constant
    ucb_c: float = 0.25  # UCB1 exploration constant for Hypervolume/Chebychev
    anneal_half_life: float = 8.0  # n_s at which beta(n_s) = 0.5
    uncertainty_gate: float = 0.15
    seed: int = 0
    checkpoint_every: int = 1  # log HV/sparsity after every N trials (1 = every trial)
    early_stop: bool = True  # stop once the archive hypervolume plateaus
    es_tol: float = 0.01  # min relative HV gain over the window to keep going (high = lenient)
    es_patience: int = 50  # trailing window, in trials, over which the gain is measured
    es_min_trials: int = 100  # never stop before this many trials have run

@dataclass
class RewardConfig:

    n_paraphrases: int = 5  # m: paraphrase variants per problem (incl. the original)
    samples_per_item: int = 3  # R (shared with SearchConfig.chance_samples in spirit)


@dataclass
class BackendConfig:

    kind: str = "openai_compatible"  # the only supported backend
    model: str = ""  # a real model id served by the proxy, e.g. "gpt-3.5-turbo"
    temperature: float = 0.7
    seed: int = 0
    base_url: str = "http://localhost:4000"  # any OpenAI-compatible endpoint, e.g. a local LiteLLM proxy
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 60.0
    realizations: int = 3
    max_concurrency: int = 8


@dataclass
class PredictorConfig:

    kind: str = "heuristic"  # "heuristic" | "gnn" (needs torch)
    device: str | None = None  # torch device for the GNN/encoder: "cpu", "cuda", or None=auto
    refit_every: int = 32  # refit after this many new replay-buffer entries
    hidden: int = 64
    # evidence floor on predictor uncertainty: floor = 1/(1 + n_train/evidence_scale).
    evidence_scale: float = 16.0
    seed: int = 0
    role_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class WandbConfig:

    enabled: bool = False  # log GNN train/test metrics to Weights & Biases
    project: str = "moo-mcts-gnn"
    run_name: str = None


@dataclass
class RunConfig:

    search: SearchConfig = field(default_factory=SearchConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    objectives: tuple[str, ...] = None
    max_workers: int = 8