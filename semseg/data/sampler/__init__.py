import logging
import random
from typing import Iterator, List

from torch.utils.data import BatchSampler


class MixedBatchSampler(BatchSampler):

    def __init__(
        self,
        sampler,
        batch_size,
        drop_last=False,
        real_size=None,
        synthetic_size=None,
        real_ratio=0.5,
        shuffle=True,
    ):
        """
        Args:
            sampler: Base sampler for the dataset (will be used only for determining total size)
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
            real_size: Number of samples in real dataset
            synthetic_size: Number of samples in synthetic dataset
            real_ratio: Ratio of real samples in each batch (0.0 to 1.0)
            shuffle: Whether to shuffle indices within each batch
        """

        super().__init__(sampler, batch_size, drop_last)

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("MixedBatchSampler")

        self.logger.info(
            f"Real size={real_size}, Synthetic size={synthetic_size}, Real ratio={real_ratio}"
        )

        self.real_size = real_size
        self.synthetic_size = synthetic_size
        self.real_ratio = real_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Calculate how many samples in each batch from each source
        self.real_per_batch = int(batch_size * real_ratio)
        self.synthetic_per_batch = batch_size - self.real_per_batch

        # Ensure at least one sample of each type if possible
        if self.real_per_batch == 0 and real_size > 0:
            self.real_per_batch = 1
            self.synthetic_per_batch = batch_size - 1

        # Total batches is determined by how many samples are available from each source
        max_real_batch = (
            real_size // self.real_per_batch
            if self.real_per_batch > 0
            else float("inf")
        )
        max_synthetic_batch = (
            synthetic_size // self.synthetic_per_batch
            if self.synthetic_per_batch > 0
            else float("inf")
        )

        self.num_batches = min(max_real_batch, max_synthetic_batch)

        # Handle drop_last (create potentially smaller final batch)
        remainder_real = (
            real_size % self.real_per_batch if self.real_per_batch > 0 else 0
        )
        remainder_synthetic = (
            synthetic_size % self.synthetic_per_batch
            if self.synthetic_per_batch > 0
            else 0
        )

        if not drop_last and (remainder_real > 0 or remainder_synthetic > 0):
            self.num_batches += 1

        if self.num_batches == 0:
            raise ValueError(
                f"Can't create mixed batches: not enough samples for the requested ratio. "
                f"Real: {real_size}, Synthetic: {synthetic_size}, "
                f"Batch size: {batch_size}, Real ratio: {real_ratio}"
            )

        self.sampler_id = f"{id(self) % 10000:04d}"
        self.logger.info(
            f"[Sampler-{self.sampler_id}] Initialized with real_size={self.real_size}, synthetic_size={self.synthetic_size}"
        )
        self.logger.info(
            f"[Sampler-{self.sampler_id}] Batch composition: {self.real_per_batch} real + {self.synthetic_per_batch} synthetic samples"
        )
        self.logger.info(
            f"[Sampler-{self.sampler_id}] Total batches: {self.num_batches}, drop_last: {self.drop_last}"
        )

    def __iter__(self):
        real_indices = list(range(self.real_size))
        synthetic_indices = list(
            range(self.real_size, self.real_size + self.synthetic_size)
        )

        real_sample = real_indices[:5] if real_indices else []
        synth_sample = synthetic_indices[:5] if synthetic_indices else []
        self.logger.info(
            f"[Sampler-{self.sampler_id}] Standard mode: {len(real_indices)} real indices (sample: {real_sample}), "
            f"{len(synthetic_indices)} synthetic indices (sample: {synth_sample})"
        )

        if self.shuffle:
            random.shuffle(real_indices)
            random.shuffle(synthetic_indices)

            real_sample_after = real_indices[:5] if real_indices else []
            synth_sample_after = synthetic_indices[:5] if synthetic_indices else []
            self.logger.info(
                f"[Sampler-{self.sampler_id}] Shuffled indices samples - Real: {real_sample_after}, Synthetic: {synth_sample_after}"
            )

        batch_count = 0
        expected_batch_size = self.real_per_batch + self.synthetic_per_batch

        for batch_idx in range(self.num_batches):
            is_last_batch = batch_idx == self.num_batches - 1

            # Get real indices for this batch
            real_batch = []
            if self.real_per_batch > 0:
                real_start = (batch_idx * self.real_per_batch) % max(
                    1, len(real_indices)
                )

                # For last batch, handle smaller size if needed
                real_to_take = (
                    min(self.real_per_batch, len(real_indices) - real_start)
                    if is_last_batch and not self.drop_last
                    else self.real_per_batch
                )

                for i in range(real_to_take):
                    idx = (real_start + i) % len(real_indices)
                    real_batch.append(real_indices[idx])

            # Get synthetic indices for this batch
            synthetic_batch = []
            if self.synthetic_per_batch > 0:
                synthetic_start = (batch_idx * self.synthetic_per_batch) % max(
                    1, len(synthetic_indices)
                )

                # For last batch, handle smaller size if needed
                synthetic_to_take = (
                    min(
                        self.synthetic_per_batch,
                        len(synthetic_indices) - synthetic_start,
                    )
                    if is_last_batch and not self.drop_last
                    else self.synthetic_per_batch
                )

                for i in range(synthetic_to_take):
                    idx = (synthetic_start + i) % len(synthetic_indices)
                    synthetic_batch.append(synthetic_indices[idx])

            # Skip empty batches
            if not real_batch and not synthetic_batch:
                continue

            # Skip last batch if drop_last and it's smaller than expected
            if (
                is_last_batch
                and self.drop_last
                and (
                    len(real_batch) < self.real_per_batch
                    or len(synthetic_batch) < self.synthetic_per_batch
                )
            ):
                self.logger.info(
                    f"[Sampler-{self.sampler_id}] Dropping last incomplete batch (real: {len(real_batch)}, synthetic: {len(synthetic_batch)})"
                )
                continue

            # Combine real and synthetic
            batch_indices = real_batch + synthetic_batch

            if self.shuffle:
                random.shuffle(batch_indices)

            # 9. Yield the final batch and count it
            batch_count += 1
            yield batch_indices

        self.logger.info(
            f"[Sampler-{self.sampler_id}] Generated {batch_count} batches in total"
        )

    def __len__(self):
        return self.num_batches
