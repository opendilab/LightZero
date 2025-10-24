# async_training_coordinator.py
"""
[PRIORZERO] Async Training Coordinator

This module implements async coordination for collect/train/eval tasks.

Key Features:
- Configurable off-policy degree to control async level
- Automatic fallback to synchronous mode (off_policy_degree=0)
- Independent async evaluation
- Thread-safe buffer access control

Author: PriorZero Team
Date: 2025-01-21
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable, Awaitable
from loguru import logger


class AsyncTrainingCoordinator:
    """
    Coordinates async execution of collect, train, and eval tasks.

    The coordinator manages the async execution based on off_policy_degree:
    - off_policy_degree = 0: Synchronous mode (collect -> train -> eval)
    - off_policy_degree > 0: Async mode with bounded lag

    The off_policy_degree controls how many batches the training can lag
    behind the collection. Higher values allow more async execution but
    increase off-policy bias.
    """

    def __init__(
        self,
        off_policy_degree: int = 0,
        enable_async_eval: bool = False,
        buffer_size: int = 10000,
        batch_size: int = 32,
    ):
        """
        Initialize AsyncTrainingCoordinator.

        Args:
            off_policy_degree: Degree of async between collect and train
                - 0: Synchronous mode
                - >0: Max number of batches train can lag behind collect
                - -1: Auto-tune based on buffer_size and batch_size
            enable_async_eval: Whether to run eval asynchronously
            buffer_size: Replay buffer size (for auto-tuning)
            batch_size: Training batch size (for auto-tuning)
        """
        self.off_policy_degree = off_policy_degree
        self.enable_async_eval = enable_async_eval
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Auto-tune off_policy_degree if set to -1
        if self.off_policy_degree == -1:
            # Auto-tune: allow lag up to 10% of buffer capacity
            self.off_policy_degree = max(1, (buffer_size // batch_size) // 10)
            logger.info(f"Auto-tuned off_policy_degree to {self.off_policy_degree}")

        # Synchronization primitives
        self._collect_count = 0  # Number of collect iterations completed
        self._train_count = 0    # Number of train iterations completed
        self._eval_task: Optional[asyncio.Task] = None

        # Locks for thread-safe access
        self._lock = asyncio.Lock()

        # Performance tracking
        self._collect_times = []
        self._train_times = []
        self._eval_times = []

        logger.info(f"AsyncTrainingCoordinator initialized:")
        logger.info(f"  - off_policy_degree: {self.off_policy_degree}")
        logger.info(f"  - enable_async_eval: {self.enable_async_eval}")
        logger.info(f"  - mode: {'SYNCHRONOUS' if self.is_synchronous else 'ASYNCHRONOUS'}")

    @property
    def is_synchronous(self) -> bool:
        """Check if coordinator is in synchronous mode."""
        return self.off_policy_degree == 0

    @property
    def collect_train_lag(self) -> int:
        """Get current lag between collect and train iterations."""
        return self._collect_count - self._train_count

    def can_train(self) -> bool:
        """
        Check if training is allowed based on off_policy_degree.

        In synchronous mode (off_policy_degree=0), training must wait for collect.
        In async mode, training can proceed as long as lag is within bounds.
        """
        if self.is_synchronous:
            # Synchronous: train only after collect
            return self._collect_count > self._train_count
        else:
            # Async: train can proceed if there's data and lag is acceptable
            # We allow training as long as there's collected data
            return self._collect_count > 0

    def can_collect(self) -> bool:
        """
        Check if collection is allowed based on off_policy_degree.

        In synchronous mode, collection must wait for train to finish.
        In async mode, collection can proceed as long as lag doesn't exceed limit.
        """
        if self.is_synchronous:
            # Synchronous: collect only after train
            return self._train_count >= self._collect_count
        else:
            # Async: collect can proceed if lag is within bounds
            lag = self.collect_train_lag
            return lag < self.off_policy_degree

    async def run_collect(
        self,
        collect_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Run collection with coordination.

        Args:
            collect_fn: Async collection function

        Returns:
            Collection result
        """
        # Wait if needed (for sync mode or if lag is too high)
        while not self.can_collect():
            logger.debug(f"Collect waiting (lag={self.collect_train_lag}, limit={self.off_policy_degree})")
            await asyncio.sleep(0.1)

        # Run collection
        start_time = time.time()
        result = await collect_fn()
        elapsed = time.time() - start_time

        # Update counter
        async with self._lock:
            self._collect_count += 1
            self._collect_times.append(elapsed)

        logger.debug(f"Collect completed in {elapsed:.2f}s (count={self._collect_count})")
        return result

    async def run_train(
        self,
        train_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Run training with coordination.

        Args:
            train_fn: Async training function

        Returns:
            Training result
        """
        # Wait if needed
        while not self.can_train():
            logger.debug(f"Train waiting (collect={self._collect_count}, train={self._train_count})")
            await asyncio.sleep(0.1)

        # Run training
        start_time = time.time()
        result = await train_fn()
        elapsed = time.time() - start_time

        # Update counter
        async with self._lock:
            self._train_count += 1
            self._train_times.append(elapsed)

        logger.debug(f"Train completed in {elapsed:.2f}s (count={self._train_count}, lag={self.collect_train_lag})")
        return result

    async def run_eval(
        self,
        eval_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Run evaluation with coordination.

        Args:
            eval_fn: Async evaluation function

        Returns:
            Evaluation result
        """
        start_time = time.time()

        if self.enable_async_eval:
            # Cancel previous eval if still running
            if self._eval_task is not None and not self._eval_task.done():
                logger.info("Cancelling previous eval task")
                self._eval_task.cancel()
                try:
                    await self._eval_task
                except asyncio.CancelledError:
                    pass

            # Run eval in background
            self._eval_task = asyncio.create_task(eval_fn())
            logger.info("Started async eval in background")

            # Return immediately (don't wait)
            return None
        else:
            # Synchronous eval
            result = await eval_fn()
            elapsed = time.time() - start_time
            self._eval_times.append(elapsed)
            logger.debug(f"Eval completed in {elapsed:.2f}s")
            return result

    async def wait_for_eval(self) -> Optional[Any]:
        """
        Wait for async eval to complete (if running).

        Returns:
            Eval result if eval was running, None otherwise
        """
        if self._eval_task is not None and not self._eval_task.done():
            logger.info("Waiting for async eval to complete...")
            try:
                result = await self._eval_task
                return result
            except asyncio.CancelledError:
                logger.warning("Eval task was cancelled")
                return None
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with timing statistics
        """
        stats = {
            'collect_count': self._collect_count,
            'train_count': self._train_count,
            'collect_train_lag': self.collect_train_lag,
            'mode': 'synchronous' if self.is_synchronous else 'asynchronous',
        }

        if self._collect_times:
            stats['collect_avg_time'] = sum(self._collect_times) / len(self._collect_times)
            stats['collect_total_time'] = sum(self._collect_times)

        if self._train_times:
            stats['train_avg_time'] = sum(self._train_times) / len(self._train_times)
            stats['train_total_time'] = sum(self._train_times)

        if self._eval_times:
            stats['eval_avg_time'] = sum(self._eval_times) / len(self._eval_times)
            stats['eval_total_time'] = sum(self._eval_times)

        return stats

    def reset_counters(self):
        """Reset all counters (useful for testing)."""
        self._collect_count = 0
        self._train_count = 0
        self._collect_times.clear()
        self._train_times.clear()
        self._eval_times.clear()
        logger.info("AsyncTrainingCoordinator counters reset")


async def run_async_training_loop(
    coordinator: AsyncTrainingCoordinator,
    collect_fn: Callable[[], Awaitable[Any]],
    train_fn: Callable[[], Awaitable[Any]],
    eval_fn: Callable[[], Awaitable[Any]],
    eval_interval: int,
    max_iterations: int,
):
    """
    Main async training loop that coordinates collect/train/eval.

    Args:
        coordinator: AsyncTrainingCoordinator instance
        collect_fn: Async collection function
        train_fn: Async training function
        eval_fn: Async evaluation function
        eval_interval: How often to run eval (in iterations)
        max_iterations: Maximum training iterations
    """
    logger.info(f"Starting async training loop (max_iter={max_iterations})")

    if coordinator.is_synchronous:
        # ========================================================================
        # SYNCHRONOUS MODE: Original serial execution
        # ========================================================================
        logger.info("Running in SYNCHRONOUS mode")

        for iteration in range(max_iterations):
            # 1. Collect
            logger.info(f"[Iter {iteration}] Collecting...")
            await coordinator.run_collect(collect_fn)

            # 2. Train
            logger.info(f"[Iter {iteration}] Training...")
            await coordinator.run_train(train_fn)

            # 3. Eval (if needed)
            if iteration % eval_interval == 0:
                logger.info(f"[Iter {iteration}] Evaluating...")
                await coordinator.run_eval(eval_fn)

    else:
        # ========================================================================
        # ASYNCHRONOUS MODE: Concurrent execution with bounded lag
        # ========================================================================
        logger.info(f"Running in ASYNCHRONOUS mode (off_policy_degree={coordinator.off_policy_degree})")

        # Create tasks for collect and train
        collect_task = None
        train_tasks = []

        iteration = 0
        while iteration < max_iterations:
            tasks_to_wait = []

            # Start collect if allowed
            if coordinator.can_collect() and (collect_task is None or collect_task.done()):
                logger.debug(f"[Iter {iteration}] Starting collect task")
                collect_task = asyncio.create_task(coordinator.run_collect(collect_fn))
                tasks_to_wait.append(collect_task)

            # Start train if allowed and there's data
            if coordinator.can_train():
                logger.debug(f"[Iter {iteration}] Starting train task")
                train_task = asyncio.create_task(coordinator.run_train(train_fn))
                train_tasks.append(train_task)
                tasks_to_wait.append(train_task)
                iteration += 1

            # Eval (if needed)
            if iteration % eval_interval == 0 and iteration > 0:
                logger.info(f"[Iter {iteration}] Triggering eval")
                await coordinator.run_eval(eval_fn)

            # Wait for at least one task to complete
            if tasks_to_wait:
                done, pending = await asyncio.wait(tasks_to_wait, return_when=asyncio.FIRST_COMPLETED)
                logger.debug(f"Tasks completed: {len(done)}, pending: {len(pending)}")
            else:
                # No tasks ready, wait a bit
                await asyncio.sleep(0.1)

            # Clean up completed train tasks
            train_tasks = [t for t in train_tasks if not t.done()]

        # Wait for all remaining tasks
        logger.info("Waiting for remaining tasks to complete...")
        if collect_task and not collect_task.done():
            await collect_task
        for task in train_tasks:
            if not task.done():
                await task

        # Wait for eval if running
        await coordinator.wait_for_eval()

    # Print statistics
    stats = coordinator.get_statistics()
    logger.info("="*80)
    logger.info("Training Loop Statistics:")
    logger.info(f"  Mode: {stats['mode']}")
    logger.info(f"  Collect count: {stats['collect_count']}")
    logger.info(f"  Train count: {stats['train_count']}")
    logger.info(f"  Final lag: {stats['collect_train_lag']}")
    if 'collect_avg_time' in stats:
        logger.info(f"  Avg collect time: {stats['collect_avg_time']:.2f}s")
    if 'train_avg_time' in stats:
        logger.info(f"  Avg train time: {stats['train_avg_time']:.2f}s")
    if 'eval_avg_time' in stats:
        logger.info(f"  Avg eval time: {stats['eval_avg_time']:.2f}s")
    logger.info("="*80)
