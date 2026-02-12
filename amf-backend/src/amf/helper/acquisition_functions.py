"""BatchBALD acquisition function implementation.

Credits to https://github.com/BlackHC/BatchBALD.
"""

"""BatchBALD acquisition function implementation.

Credits to https://github.com/BlackHC/BatchBALD.
"""

__all__ = [
    "CandidateBatch",
    "CandidateBatch",
    "compute_conditional_entropy",
    "get_batchbald_batch",
]


from dataclasses import dataclass

import torch
from toma import toma
from tqdm.auto import tqdm

import amf.helper.joint_entropy


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """Return the conditional entropy for each sample.

    Args:
        log_probs_N_K_C: Log class probabilities with shape (N, K, C) where N is the
            number of samples, K is the number of MC Dropout samples, and C is the
            number of classes.

    Returns: Conditional entropy for each sample, shape (N,).
    """
    N, K, _ = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)  # type: ignore[untyped-decorator]
    def compute(log_probs_n_K_C: torch.Tensor, start: int, end: int) -> None:
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


@dataclass
class CandidateBatch:
    """A batch to be labelled in BatchBALD."""

    scores: list[float]
    indices: list[int]


def get_batchbald_batch(
    log_probs_N_K_C: torch.Tensor,
    batch_size: int,
    num_samples: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> CandidateBatch:
    """Return a batch of the most uncertain samples using BatchBALD.

    Args:
        log_probs_N_K_C: Log class probabilities with shape (N, K, C) where N is the
            number of samples, K is the number of MC Dropout samples, and C is the
            number of classes.
        batch_size: Number of samples to select for labelling.
        num_samples: Number of MC Dropout samples.
        dtype: Desired data type of computations.
        device: Desired device of computations.

    Returns: Selected batch of samples to be labelled.
    """
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices: list[int] = []
    candidate_scores: list[float] = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = amf.helper.joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(
                log_probs_N_K_C[latest_index : latest_index + 1]
            )

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    return CandidateBatch(candidate_scores, candidate_indices)
