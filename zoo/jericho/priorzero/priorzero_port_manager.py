"""
Port Manager for Multi-Instance Training

Automatically finds available ports for DeepSpeed and Ray
to enable running multiple training instances on the same machine.

Author: PriorZero Team
Date: 2025-12-18
"""

import os
import socket
import random
from typing import Optional, Tuple


def find_free_port(start_port: int = 29500, end_port: int = 40000, max_attempts: int = 100) -> int:
    """
    Find a free port in the given range.

    Args:
        start_port: Start of port range to search
        end_port: End of port range to search
        max_attempts: Maximum number of ports to try

    Returns:
        Free port number

    Raises:
        RuntimeError: If no free port found after max_attempts
    """
    attempts = 0
    checked_ports = set()

    while attempts < max_attempts:
        # Random port in range to avoid sequential conflicts
        port = random.randint(start_port, end_port)

        # Skip if already checked
        if port in checked_ports:
            continue
        checked_ports.add(port)

        # Try to bind to port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            # Port is in use, try next
            attempts += 1
            continue
        finally:
            try:
                sock.close()
            except:
                pass

    raise RuntimeError(
        f"Could not find free port after {max_attempts} attempts in range [{start_port}, {end_port}]"
    )


def setup_distributed_ports(
    master_port: Optional[int] = None,
    ray_port: Optional[int] = None,
    auto_find: bool = True
) -> Tuple[int, int]:
    """
    Setup ports for distributed training (DeepSpeed + Ray).

    This function ensures that:
    1. Each training instance uses unique ports
    2. Ports don't conflict between DeepSpeed and Ray
    3. Graceful fallback if specified ports are occupied

    Args:
        master_port: Preferred port for DeepSpeed (default: auto-find)
        ray_port: Preferred port for Ray dashboard (default: auto-find)
        auto_find: If True, automatically find free ports if preferred ones are taken

    Returns:
        Tuple of (deepspeed_port, ray_port)
    """

    # DeepSpeed master port
    if master_port is None or (auto_find and not is_port_free(master_port)):
        master_port = find_free_port(start_port=29500, end_port=30500)
        print(f"[Port Manager] DeepSpeed master port: {master_port}")
    else:
        print(f"[Port Manager] Using specified DeepSpeed port: {master_port}")

    # Ray dashboard port (avoid DeepSpeed port range)
    if ray_port is None or (auto_find and not is_port_free(ray_port)):
        ray_port = find_free_port(start_port=8265, end_port=9000)
        print(f"[Port Manager] Ray dashboard port: {ray_port}")
    else:
        print(f"[Port Manager] Using specified Ray port: {ray_port}")

    # Set environment variables for DeepSpeed
    os.environ['MASTER_PORT'] = str(master_port)
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'

    return master_port, ray_port


def is_port_free(port: int) -> bool:
    """
    Check if a port is free.

    Args:
        port: Port number to check

    Returns:
        True if port is free, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('', port))
        sock.close()
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except:
            pass


def get_available_port_range(num_ports: int = 10, start_port: int = 29500) -> list:
    """
    Get a list of available ports.
    Useful for allocating ports for multiple workers.

    Args:
        num_ports: Number of ports needed
        start_port: Start searching from this port

    Returns:
        List of available port numbers
    """
    ports = []
    current_port = start_port

    while len(ports) < num_ports and current_port < 65535:
        if is_port_free(current_port):
            ports.append(current_port)
        current_port += 1

    if len(ports) < num_ports:
        raise RuntimeError(
            f"Could only find {len(ports)} free ports, but {num_ports} were requested"
        )

    return ports


def cleanup_distributed_env():
    """
    Clean up distributed training environment variables.
    Call this before starting a new training instance.
    """
    env_vars_to_remove = [
        'MASTER_PORT',
        'MASTER_ADDR',
        'RANK',
        'WORLD_SIZE',
        'LOCAL_RANK',
        'LOCAL_WORLD_SIZE'
    ]

    for var in env_vars_to_remove:
        if var in os.environ:
            del os.environ[var]
            print(f"[Port Manager] Removed environment variable: {var}")


# Convenience function for the main training script
def auto_setup_ports_for_training(
    instance_id: Optional[int] = None,
    verbose: bool = True
) -> dict:
    """
    Automatically setup all ports needed for training.
    This is the main entry point for training scripts.

    Args:
        instance_id: Optional instance ID to use predictable ports
        verbose: Whether to print setup information

    Returns:
        Dictionary with port assignments
    """
    if verbose:
        print("\n" + "="*70)
        print("Auto Port Setup for Multi-Instance Training")
        print("="*70)

    # If instance_id is provided, try to use predictable ports
    if instance_id is not None:
        base_port = 29500 + (instance_id * 100)
        ray_base_port = 8265 + instance_id
        if verbose:
            print(f"Instance ID: {instance_id}")
            print(f"Base ports: DeepSpeed={base_port}, Ray={ray_base_port}")
    else:
        base_port = None
        ray_base_port = None
        if verbose:
            print("Instance ID: Auto (will find any available ports)")

    # Setup ports
    master_port, ray_port = setup_distributed_ports(
        master_port=base_port,
        ray_port=ray_base_port,
        auto_find=True
    )

    port_config = {
        'master_port': master_port,
        'ray_port': ray_port,
        'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
    }

    if verbose:
        print(f"\nPort Configuration:")
        print(f"  • DeepSpeed Master: {master_port}")
        print(f"  • Ray Dashboard:    {ray_port}")
        print(f"  • Master Address:   {port_config['master_addr']}")
        print("="*70 + "\n")

    return port_config


# Example usage
if __name__ == "__main__":
    print("Testing Port Manager...")

    # Test 1: Find a single free port
    print("\nTest 1: Finding a free port...")
    port = find_free_port()
    print(f"  ✓ Found free port: {port}")

    # Test 2: Auto setup for training
    print("\nTest 2: Auto setup for training instance 0...")
    config1 = auto_setup_ports_for_training(instance_id=0)

    print("\nTest 3: Auto setup for training instance 1...")
    config2 = auto_setup_ports_for_training(instance_id=1)

    # Test 4: Get port range
    print("\nTest 4: Get range of 5 available ports...")
    ports = get_available_port_range(num_ports=5)
    print(f"  ✓ Available ports: {ports}")

    print("\n✓ All tests passed!")
