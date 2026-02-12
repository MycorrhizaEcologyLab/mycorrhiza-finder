"""Functions for estimating joint entropy of multiple discrete variables.

This is relevant for active learning with BatchBALD.

Adapted from https://github.com/BlackHC/BatchBALD.
"""

from abc import ABC, abstractmethod
from typing import cast

import torch
from toma import toma
from tqdm.auto import tqdm


class JointEntropy(ABC):
    """Joint entropy base class.

    Defines the interface that specific joint entropy implementations must follow.
    """

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute entropy."""
        pass

    # TODO solve the mypy override errors properly
    # - overrides should not have a different signature
    @abstractmethod
    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "JointEntropy":
        """Expand the joint entropy to include more terms."""
        pass

    @abstractmethod
    def compute_batch(  # type: ignore[no-untyped-def]
        self, log_probs_B_K_C: torch.Tensor, output_entropies_B=None
    ) -> torch.Tensor:
        """Compute the joint entropy of the added variables together with the batch.

        Computes the joint entropy of the added variables with each of the variables in
        the provided batch probabilities in turn.
        """
        pass


class ExactJointEntropy(JointEntropy):
    """Exact joint entropy implementation.

    Uses exact computation to calculate the joint entropy of multiple discrete
    variables.
    """

    joint_probs_M_K: torch.Tensor

    def __init__(self, joint_probs_M_K: torch.Tensor):
        """Initialise exact joint entropy with given joint probabilities.

        Args:
            joint_probs_M_K: Joint probabilities of M joint configurations over K
                samples.
        """
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(
        K: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> "ExactJointEntropy":
        """Create empty ExactJointEntropy for K samples.

        Initialised with only one configuration, with probability 1 for all samples.

        Args:
            K: Number of samples.
            device: Torch device.
            dtype: Torch data type.

        Returns: New empty ExactJointEntropy instance.
        """
        return ExactJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    def compute(self) -> torch.Tensor:
        """Compute and return joint entropy."""
        probs_M = torch.mean(self.joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(probs_M) * probs_M
        return torch.sum(nats_M)

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "ExactJointEntropy":
        """Add more variables to the joint entropy.

        Args:
            log_probs_N_K_C: Log probabilities of the new variables to add. Shape
                (N, K, C) where N is the number of new variables, K is the number of
                samples, and C is the number of classes.

        Returns: Self, with updated joint probabilities.
        """
        if self.joint_probs_M_K.shape[1] != log_probs_N_K_C.shape[1]:
            raise ValueError(
                f"Cannot add variables with K={log_probs_N_K_C.shape[1]} to "
                f"ExactJointEntropy with K={self.joint_probs_M_K.shape[1]}."
            )

        N, K, _ = log_probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.t()[:, :, None]

        probs_N_K_C = log_probs_N_K_C.exp()

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :].to(
                joint_probs_K_M_1, non_blocking=True
            )
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).t()
        return self

    def compute_batch(
        self,
        log_probs_B_K_C: torch.Tensor,
        output_entropies_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint entropy of added variables with the batch.

        Computes the joint entropy of the added variables with each of the variables in
        the provided batch probabilities in turn.

        Args:
            log_probs_B_K_C: Log probabilities of the batch variables. Shape
                (B, K, C) where B is the batch size, K is the number of samples, and C
                is the number of classes.
            output_entropies_B: Optional preallocated tensor to store the output
                entropies. If None, a new tensor will be created.

        Returns: Tensor of shape (B,) containing the joint entropies.
        """
        if self.joint_probs_M_K.shape[1] != log_probs_B_K_C.shape[1]:
            raise ValueError(
                f"Cannot compute batch with K={log_probs_B_K_C.shape[1]} for "
                f"ExactJointEntropy with K={self.joint_probs_M_K.shape[1]}."
            )

        B, K, C = log_probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(
                B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device
            )

        pbar = tqdm(total=B, desc="ExactJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)  # type: ignore[untyped-decorator]
        def chunked_joint_entropy(
            chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int
        ) -> None:
            chunked_probs_b_K_C = chunked_log_probs_b_K_C.exp()
            b = chunked_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.joint_probs_M_K.dtype,
                device=self.joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.joint_probs_M_K,
                    chunked_probs_b_K_C[i].to(self.joint_probs_M_K, non_blocking=True),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C, dim=(1, 2)),
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


def batch_multi_choices(probs_b_C: torch.Tensor, M: int) -> torch.Tensor:
    """Sample M choices from categorical distributions defined by probs_b_C.

    Args:
        probs_b_C: Batch of categorical distributions. Shape: (..., C) where ...
            represents any number of batch dimensions and C is the number of classes.
        M: Number of samples to draw from each categorical distribution.

    Returns: Sampled category indices. Shape: (..., M) where ... matches
            the batch dimensions of probs_b_C, and M is the number of samples.
            Each value is an integer in [0, C) representing a sampled category.
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))

    # samples: Ni... x draw_per_xx
    choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)

    return cast(torch.Tensor, choices.reshape(*probs_b_C.shape[:-1], M))


def gather_expand(data: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Select values from data using indices, with automatic broadcasting.

    Selects elements from the `tensor` corresponding to indices in `index` along the
    specified dimension `dim`. Extends torch.gather by automatically broadcasting data
    and index tensors to compatible shapes.

    Args:
        data: Source tensor to select values from. Shape: (..., D_dim, ...)
        dim: Dimension along which to index into data.
        index: Indices specifying which elements to select along dim.
            Shape: (..., I_dim, ...) where each dimension (except dim) must be
            either 1, equal to data's corresponding dimension, or larger (in which
            case data is expanded). Values must be integers in [0, D_dim).

    Returns: Selected values with shape (..., I_dim, ...) where each dimension
        (except dim) has size max(data.shape[d], index.shape[d]), and the
        dimension at dim has size index.shape[dim].
    """
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape, strict=True)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)


class SampledJointEntropy(JointEntropy):
    """Sampled joint entropy implementation.

    Uses random sampling to estimate the joint entropy of multiple discrete variables.
    """

    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor):
        """Construct SampledJointEntropy with given sampled joint probabilities.

        Args:
            sampled_joint_probs_M_K: Sampled joint probabilities of M joint
                configurations over K samples.
        """
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def empty(
        K: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> "SampledJointEntropy":
        """Construct empty SampledJointEntropy for K samples.

        Initialised with only one configuration, with probability 1 for all samples.

        Args:
            K: Number of samples.
            device: Torch device.
            dtype: Torch data type.
        """
        return SampledJointEntropy(torch.ones((1, K), device=device, dtype=dtype))

    @staticmethod
    def sample(probs_N_K_C: torch.Tensor, M: int) -> "SampledJointEntropy":
        """Sample joint probabilities from given categorical distributions.

        Args:
            probs_N_K_C: Probabilities of the categorical distributions to sample
                from. Shape: (N, K, C) where N is the number of variables, K is the
                number of samples, and C is the number of classes.
            M: Number of joint samples to draw.

        Returns: SampledJointEntropy instance with sampled joint probabilities of
            shape (M, K).
        """
        K = probs_N_K_C.shape[1]

        # S: num of samples per w
        S = M // K

        choices_N_K_S = batch_multi_choices(probs_N_K_C, S).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = gather_expand(
            expanded_probs_N_K_1_C, dim=-1, index=expanded_choices_N_1_K_S
        )
        # exp sum log seems necessary to avoid 0s?
        probs_K_K_S = torch.exp(
            torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False)
        )
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.t()
        return SampledJointEntropy(samples_M_K)

    def compute(self) -> torch.Tensor:
        """Compute and return joint entropy."""
        sampled_joint_probs_M = torch.mean(
            self.sampled_joint_probs_M_K, dim=1, keepdim=False
        )
        nats_M = -torch.log(sampled_joint_probs_M)
        return torch.mean(nats_M)

    def add_variables(  # type: ignore[override]
        self, log_probs_N_K_C: torch.Tensor, M2: int
    ) -> "SampledJointEntropy":
        """Add more variables to the joint entropy.

        Args:
            log_probs_N_K_C: Log probabilities of the new variables to add. Shape
                (N, K, C) where N is the number of new variables, K is the number of
                samples, and C is the number of classes.
            M2: Number of joint samples to draw for the new variables.

        Returns: Self, with updated sampled joint probabilities.
        """
        K = self.sampled_joint_probs_M_K.shape[1]
        if K != log_probs_N_K_C.shape[1]:
            raise ValueError(
                f"Cannot add variables with K={log_probs_N_K_C.shape[1]} to "
                f"SampledJointEntropy with K={K}."
            )

        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]

        new_sample_M2_K = self.sample(log_probs_N_K_C.exp(), M2).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]

        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))

        self.sampled_joint_probs_M_K = merged_sample_K_M.t()

        return self

    def compute_batch(
        self,
        log_probs_B_K_C: torch.Tensor,
        output_entropies_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint entropy of added variables with the batch.

        Computes the joint entropy of the added variables with each of the variables in
        the provided batch probabilities in turn.

        Args:
            log_probs_B_K_C: Log probabilities of the batch variables. Shape
                (B, K, C) where B is the batch size, K is the number of samples, and C
                is the number of classes.
            output_entropies_B: Optional preallocated tensor to store the output
                entropies. If None, a new tensor will be created.

        Returns: Tensor of shape (B,) containing the joint entropies.
        """
        if self.sampled_joint_probs_M_K.shape[1] != log_probs_B_K_C.shape[1]:
            raise ValueError(
                f"Cannot compute batch with K={log_probs_B_K_C.shape[1]} for "
                f"SampledJointEntropy with K={self.sampled_joint_probs_M_K.shape[1]}."
            )

        B, K, C = log_probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(
                B, dtype=log_probs_B_K_C.dtype, device=log_probs_B_K_C.device
            )

        pbar = tqdm(total=B, desc="SampledJointEntropy.compute_batch", leave=False)

        @toma.execute.chunked(log_probs_B_K_C, initial_step=1024, dimension=0)  # type: ignore[untyped-decorator]
        def chunked_joint_entropy(
            chunked_log_probs_b_K_C: torch.Tensor, start: int, end: int
        ) -> None:
            b = chunked_log_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device,
            )
            for i in range(b):
                torch.matmul(
                    self.sampled_joint_probs_M_K,
                    chunked_log_probs_b_K_C[i]
                    .to(self.sampled_joint_probs_M_K, non_blocking=True)
                    .exp(),
                    out=probs_b_M_C[i],
                )
            probs_b_M_C /= K

            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1, keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1, dim=(1, 2))
                / M,
                non_blocking=True,
            )

            pbar.update(end - start)

        pbar.close()

        return output_entropies_B


class DynamicJointEntropy(JointEntropy):
    """Dynamic joint entropy implementation.

    Dynamic joint entropy is a measure of uncertainty across multiple predictions that
    grows incrementally as new data points are added. Unlike static joint entropy which
    is computed over a fixed set of items, dynamic joint entropy allows you to add items
    one at a time and efficiently update the entropy calculation without recomputing
    from scratch.

    This is useful in active learning scenarios where you want to measure the joint
    uncertainty of a growing batch of candidate samples to select the most informative
    subset.
    """

    inner: JointEntropy
    log_probs_max_N_K_C: torch.Tensor
    N: int
    M: int

    def __init__(
        self,
        M: int,
        max_N: int,
        K: int,
        C: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialise DynamicJointEntropy.

        Args:
            M: Number of samples drawn from each categorical distribution (for Monte
                Carlo estimation of entropy).
            max_N: Maximum number of data points that can be added dynamically.
                Pre-allocates storage for efficiency.
            K: Number of models or ensemble members making predictions.
            C: Number of classes in the classification task.
            dtype: Torch data type.
            device: Torch device.
        """
        self.M = M
        self.N = 0
        self.max_N = max_N

        self.inner = ExactJointEntropy.empty(K, dtype=dtype, device=device)
        self.log_probs_max_N_K_C = torch.empty(
            (max_N, K, C), dtype=dtype, device=device
        )

    def add_variables(self, log_probs_N_K_C: torch.Tensor) -> "DynamicJointEntropy":
        """Add more variables to the joint entropy calculation.

        Args:
            log_probs_N_K_C: Log probabilities of the new variables to add. Shape
                (N, K, C) where N is the number of new variables, K is the number of
                samples, and C is the number of classes.

        Returns: Self, with updated joint entropy calculation.
        """
        C = self.log_probs_max_N_K_C.shape[2]
        add_N = log_probs_N_K_C.shape[0]

        if self.log_probs_max_N_K_C.shape[0] < self.N + add_N:
            raise ValueError(
                f"Cannot add {add_N} variables to DynamicJointEntropy with "
                f"max_N={self.max_N} and current N={self.N}."
            )
        if self.log_probs_max_N_K_C.shape[1] != log_probs_N_K_C.shape[1]:
            raise ValueError(
                f"Cannot add variables with K={log_probs_N_K_C.shape[1]} to "
                f"DynamicJointEntropy with K={self.log_probs_max_N_K_C.shape[1]}."
            )

        self.log_probs_max_N_K_C[self.N : self.N + add_N] = log_probs_N_K_C
        self.N += add_N

        num_exact_samples = C**self.N
        if num_exact_samples > self.M:
            self.inner = SampledJointEntropy.sample(
                self.log_probs_max_N_K_C[: self.N].exp(), self.M
            )
        else:
            self.inner.add_variables(log_probs_N_K_C)

        return self

    def compute(self) -> torch.Tensor:
        """Compute and return joint entropy."""
        return self.inner.compute()

    def compute_batch(
        self,
        log_probs_B_K_C: torch.Tensor,
        output_entropies_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute joint entropy of added variables with the batch.

        Computes the joint entropy of the added variables with each of the variables in
        the provided batch probabilities in turn.

        Args:
            log_probs_B_K_C: Log probabilities of the batch variables. Shape
                (B, K, C) where B is the batch size, K is the number of samples, and C
                is the number of classes.
            output_entropies_B: Optional preallocated tensor to store the output
                entropies. If None, a new tensor will be created.

        Returns: Tensor of shape (B,) containing the joint entropies.
        """
        return self.inner.compute_batch(log_probs_B_K_C, output_entropies_B)
