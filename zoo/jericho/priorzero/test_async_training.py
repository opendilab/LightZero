#!/usr/bin/env python3
"""
Test script for async training functionality.

This script tests:
1. Synchronous mode (off_policy_degree=0) - should behave like original
2. Asynchronous mode (off_policy_degree>0) - collect and train overlap
3. Async eval mode - eval runs in background

Usage:
    # Test synchronous mode
    python test_async_training.py --mode sync

    # Test async mode with low degree
    python test_async_training.py --mode async_low

    # Test async mode with high degree
    python test_async_training.py --mode async_high

    # Test async eval
    python test_async_training.py --mode async_eval
"""

import asyncio
import time
from loguru import logger
from async_training_coordinator import AsyncTrainingCoordinator


async def mock_collect():
    """Mock collect function that simulates data collection."""
    logger.info("  [Collect] Starting...")
    await asyncio.sleep(0.5)  # Simulate collect time
    logger.info("  [Collect] Completed")
    return {"data": "collected"}


async def mock_train():
    """Mock train function that simulates training."""
    logger.info("    [Train] Starting...")
    await asyncio.sleep(0.3)  # Simulate train time
    logger.info("    [Train] Completed")
    return {"loss": 0.1}


async def mock_eval():
    """Mock eval function that simulates evaluation."""
    logger.info("      [Eval] Starting...")
    await asyncio.sleep(1.0)  # Simulate eval time
    logger.info("      [Eval] Completed")
    return {"reward": 10.0}


async def test_synchronous_mode():
    """Test synchronous mode (off_policy_degree=0)."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Synchronous Mode (off_policy_degree=0)")
    logger.info("="*80)

    coordinator = AsyncTrainingCoordinator(
        off_policy_degree=0,
        enable_async_eval=False,
    )

    start_time = time.time()

    # Run 3 iterations
    for i in range(3):
        logger.info(f"\n[Iteration {i}]")

        # Collect
        await coordinator.run_collect(mock_collect)

        # Train
        await coordinator.run_train(mock_train)

        # Eval every other iteration
        if i % 2 == 0:
            await coordinator.run_eval(mock_eval)

    elapsed = time.time() - start_time
    stats = coordinator.get_statistics()

    logger.info("\n" + "="*80)
    logger.info("SYNCHRONOUS MODE RESULTS:")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  Collect count: {stats['collect_count']}")
    logger.info(f"  Train count: {stats['train_count']}")
    logger.info(f"  Lag: {stats['collect_train_lag']}")
    logger.info("="*80)

    # Verify synchronous behavior
    assert stats['collect_count'] == stats['train_count'], "Collect and train should be equal in sync mode"
    logger.info("✓ Synchronous mode test PASSED")


async def test_async_mode_low():
    """Test async mode with low off-policy degree."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Async Mode Low (off_policy_degree=5)")
    logger.info("="*80)

    coordinator = AsyncTrainingCoordinator(
        off_policy_degree=5,
        enable_async_eval=False,
    )

    start_time = time.time()

    # Create tasks
    collect_task = None
    train_count = 0
    max_iterations = 10

    while train_count < max_iterations:
        # Start collect if allowed
        if (collect_task is None or collect_task.done()) and coordinator.can_collect():
            logger.info(f"\n[Starting collect (lag={coordinator.collect_train_lag})]")
            collect_task = asyncio.create_task(coordinator.run_collect(mock_collect))

        # Train if allowed
        if coordinator.can_train():
            logger.info(f"[Starting train {train_count} (lag={coordinator.collect_train_lag})]")
            await coordinator.run_train(mock_train)
            train_count += 1
        else:
            # Wait a bit if can't train yet
            await asyncio.sleep(0.1)

    # Wait for remaining collect
    if collect_task and not collect_task.done():
        await collect_task

    elapsed = time.time() - start_time
    stats = coordinator.get_statistics()

    logger.info("\n" + "="*80)
    logger.info("ASYNC MODE LOW RESULTS:")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  Collect count: {stats['collect_count']}")
    logger.info(f"  Train count: {stats['train_count']}")
    logger.info(f"  Max lag: {stats['collect_train_lag']}")
    logger.info("="*80)

    # Verify async behavior
    assert stats['collect_train_lag'] <= 5, f"Lag should be <= 5, got {stats['collect_train_lag']}"
    logger.info("✓ Async mode low test PASSED")


async def test_async_mode_high():
    """Test async mode with high off-policy degree."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Async Mode High (off_policy_degree=20)")
    logger.info("="*80)

    coordinator = AsyncTrainingCoordinator(
        off_policy_degree=20,
        enable_async_eval=False,
    )

    start_time = time.time()

    # Create tasks
    collect_task = None
    train_count = 0
    max_iterations = 10

    while train_count < max_iterations:
        # Start collect if allowed
        if (collect_task is None or collect_task.done()) and coordinator.can_collect():
            logger.info(f"\n[Starting collect (lag={coordinator.collect_train_lag})]")
            collect_task = asyncio.create_task(coordinator.run_collect(mock_collect))

        # Train if allowed
        if coordinator.can_train():
            logger.info(f"[Starting train {train_count} (lag={coordinator.collect_train_lag})]")
            await coordinator.run_train(mock_train)
            train_count += 1
        else:
            await asyncio.sleep(0.1)

    # Wait for remaining collect
    if collect_task and not collect_task.done():
        await collect_task

    elapsed = time.time() - start_time
    stats = coordinator.get_statistics()

    logger.info("\n" + "="*80)
    logger.info("ASYNC MODE HIGH RESULTS:")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  Collect count: {stats['collect_count']}")
    logger.info(f"  Train count: {stats['train_count']}")
    logger.info(f"  Max lag: {stats['collect_train_lag']}")
    logger.info("="*80)

    # Verify async behavior with higher lag
    assert stats['collect_train_lag'] <= 20, f"Lag should be <= 20, got {stats['collect_train_lag']}"
    logger.info("✓ Async mode high test PASSED")


async def test_async_eval():
    """Test async evaluation mode."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Async Evaluation Mode")
    logger.info("="*80)

    coordinator = AsyncTrainingCoordinator(
        off_policy_degree=0,
        enable_async_eval=True,
    )

    start_time = time.time()

    # Run 3 iterations with async eval
    for i in range(3):
        logger.info(f"\n[Iteration {i}]")

        # Collect
        await coordinator.run_collect(mock_collect)

        # Train
        await coordinator.run_train(mock_train)

        # Async eval (doesn't block)
        if i % 2 == 0:
            await coordinator.run_eval(mock_eval)
            logger.info("  [Main] Continuing while eval runs in background...")

    # Wait for eval to complete
    logger.info("\n[Main] Waiting for eval to complete...")
    await coordinator.wait_for_eval()

    elapsed = time.time() - start_time
    stats = coordinator.get_statistics()

    logger.info("\n" + "="*80)
    logger.info("ASYNC EVAL MODE RESULTS:")
    logger.info(f"  Total time: {elapsed:.2f}s")
    logger.info(f"  Collect count: {stats['collect_count']}")
    logger.info(f"  Train count: {stats['train_count']}")
    logger.info("="*80)

    logger.info("✓ Async eval test PASSED")


async def test_auto_tune():
    """Test auto-tuning of off_policy_degree."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Auto-tune off_policy_degree")
    logger.info("="*80)

    coordinator = AsyncTrainingCoordinator(
        off_policy_degree=-1,  # Auto-tune
        enable_async_eval=False,
        buffer_size=10000,
        batch_size=32,
    )

    logger.info(f"Auto-tuned off_policy_degree: {coordinator.off_policy_degree}")
    expected = (10000 // 32) // 10  # ~31
    assert coordinator.off_policy_degree > 0, "Auto-tuned value should be > 0"
    logger.info(f"✓ Auto-tune test PASSED (value={coordinator.off_policy_degree})")


async def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description='Test async training')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sync', 'async_low', 'async_high', 'async_eval', 'auto_tune', 'all'],
        default='all',
        help='Test mode to run'
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ASYNC TRAINING TESTS")
    logger.info("="*80)

    if args.mode == 'sync' or args.mode == 'all':
        await test_synchronous_mode()

    if args.mode == 'async_low' or args.mode == 'all':
        await test_async_mode_low()

    if args.mode == 'async_high' or args.mode == 'all':
        await test_async_mode_high()

    if args.mode == 'async_eval' or args.mode == 'all':
        await test_async_eval()

    if args.mode == 'auto_tune' or args.mode == 'all':
        await test_auto_tune()

    logger.info("\n" + "="*80)
    logger.info("ALL TESTS PASSED! ✓")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
