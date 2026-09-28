"""Shared driver helpers: backend construction (real OpenAI-compatible backend)."""
from ..config import BackendConfig
from ..objectives import ObjectiveSpec
from ..backends.task_profile import TaskProfile
from ..backends.openai_compatible import OpenAICompatibleProposer, OpenAICompatibleExecutor, make_client
from ..valuation.critic import HeuristicPrior


def build_backends(bcfg: BackendConfig, spec: ObjectiveSpec, task_profile: TaskProfile = None):
    """Return (proposer, make_executor, critic) for the configured backend.
    """
    if bcfg.kind == "openai_compatible":

        client = make_client(bcfg)
        operators = task_profile.operators if task_profile is not None else None
        proposer = OpenAICompatibleProposer(
            client, 
            spec, 
            n_realizations=bcfg.realizations, 
            operators=operators,
            task_profile=task_profile,
        )
        critic = HeuristicPrior(spec)

        def make_executor(item_seed: int):
            return OpenAICompatibleExecutor(client, seed=item_seed, profile=task_profile)

        return proposer, make_executor, critic

    raise ValueError(f"unknown backend kind {bcfg.kind!r}")
