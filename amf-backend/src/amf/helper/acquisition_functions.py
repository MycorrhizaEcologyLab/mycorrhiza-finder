# Credits to https://github.com/BlackHC/BatchBALD
__all__ = [
    "compute_conditional_entropy",
    "compute_entropy",
    "CandidateBatch",
    "get_batchbald_batch",
]


import math
from dataclasses import dataclass
from typing import List

import torch
from toma import toma
from tqdm.auto import tqdm

import amf.helper.joint_entropy


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)  # type: ignore[misc]
    def compute(log_probs_n_K_C: torch.Tensor, start: int, end: int) -> None:
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)  # type: ignore[misc]
    def compute(log_probs_n_K_C: torch.Tensor, start: int, end: int) -> None:
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - torch.log(
            torch.tensor(K)
        )
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def get_batchbald_batch(
    log_probs_N_K_C: torch.Tensor,
    batch_size: int,
    num_samples: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> CandidateBatch:
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
