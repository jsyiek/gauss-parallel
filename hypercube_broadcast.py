#!/usr/bin/env python3
"""
Hypercube Broadcasting Implementation using MPI4Py

This module demonstrates hypercube broadcasting algorithm that can broadcast
data from any root process to all other processes in O(log n) steps.

The hypercube algorithm works by:
1. Organizing processes in a logical hypercube topology
2. Broadcasting data through hypercube dimensions using XOR operations
3. Each step doubles the number of processes that have the data

Can be imported as a module or run standalone with argument parsing.
"""

import math
import time
import argparse
from mpi4py import MPI


def hypercube_broadcast(data, root, comm, verbose=False):
    """
    Broadcast data from root process to all processes using hypercube algorithm.
    
    Args:
        data: Data to broadcast (only meaningful on root process)
        root: Root process rank that initially has the data
        comm: MPI communicator
        verbose: If True, print detailed step information
        
    Returns:
        The broadcasted data (received on all processes)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if verbose and rank == 0:
        print(f"Starting hypercube broadcast from root={root}, size={size}")
    
    # Handle single process case
    if size == 1:
        return data
    
    # Initialize: only root has data, others have None
    if rank == root:
        local_data = data
    else:
        local_data = None
    
    # Calculate number of hypercube dimensions needed
    dimensions = int(math.ceil(math.log2(size)))
    
    if verbose and rank == 0:
        print(f"Hypercube dimensions: {dimensions}")
    
    # Transform ranks relative to root for hypercube calculations
    # This allows any process to be the root
    transformed_rank = (rank - root) % size
    
    # Hypercube broadcasting algorithm
    for dim in range(dimensions):
        # Determine if this process should send in this dimension
        if transformed_rank < (1 << dim):
            # This process has data and should send
            neighbor_transformed = transformed_rank ^ (1 << dim)
            neighbor_rank = (neighbor_transformed + root) % size
            
            # Only send if neighbor exists (for non-power-of-2 sizes)
            if neighbor_rank < size:
                if verbose:
                    print(f"Step {dim+1}: Rank {rank} -> Rank {neighbor_rank}")
                comm.send(local_data, dest=neighbor_rank, tag=dim)
        else:
            # Check if this process should receive in this dimension
            sender_transformed = transformed_rank ^ (1 << dim)
            if sender_transformed < (1 << dim):
                sender_rank = (sender_transformed + root) % size
                
                if sender_rank < size:
                    if verbose:
                        print(f"Step {dim+1}: Rank {rank} <- Rank {sender_rank}")
                    local_data = comm.recv(source=sender_rank, tag=dim)
        
        # Synchronize all processes before next step
        comm.Barrier()
    
    if verbose and rank == 0:
        print("Hypercube broadcast complete")
    
    return local_data


def main():
    """Main function when run as standalone script."""
    parser = argparse.ArgumentParser(
        description="Demonstrate hypercube broadcasting with MPI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--value", 
        type=str, 
        default=None,
        help="Value to broadcast (default: current timestamp)"
    )
    parser.add_argument(
        "--root", 
        type=int, 
        default=0,
        help="Root process rank to broadcast from"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed step-by-step progress"
    )
    
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Validate root rank
    if args.root >= size or args.root < 0:
        if rank == 0:
            print(f"Error: Root rank {args.root} is invalid for {size} processes")
        return
    
    # Prepare data to broadcast
    if rank == args.root:
        if args.value is not None:
            broadcast_data = args.value
        else:
            broadcast_data = f"Timestamp: {time.time()}"
        
        print(f"Root process {rank} broadcasting: {broadcast_data}")
    else:
        broadcast_data = None
    
    # Perform hypercube broadcast
    start_time = time.time()
    result = hypercube_broadcast(broadcast_data, args.root, comm, args.verbose)
    end_time = time.time()
    
    # Show results
    print(f"Process {rank} received: {result}")
    
    # Root process shows timing summary
    if rank == args.root:
        elapsed = end_time - start_time
        steps = int(math.ceil(math.log2(size))) if size > 1 else 0
        print(f"Broadcast completed in {elapsed:.6f} seconds using {steps} steps")
        print(f"Theoretical minimum steps for {size} processes: {steps}")


if __name__ == "__main__":
    main()