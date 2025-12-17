#!/usr/bin/env python3
"""
Parallel Gauss-Seidel Algorithm for Hypercube Architecture

This module implements the parallel Gauss-Seidel iterative method for solving
systems of linear equations Ax = b using a hypercube communication pattern.

Algorithm:
- For each iteration and equation i (in order):
  - Process owning equation i computes x_i(t+1) using:
    - x_0(t+1), ..., x_{i-1}(t+1) (already computed this iteration)
    - x_i(t), ..., x_n(t) (from previous iteration)
  - Broadcast x_i(t+1) to all processes via hypercube
  - All processes update their local x vector
- Continue until convergence or max iterations

Can be imported as a module or run standalone with demo matrices.
"""

import numpy as np
import argparse
from mpi4py import MPI
from hypercube_broadcast import hypercube_broadcast


def parallel_gauss_seidel(A, b, x0=None, max_iter=1000, tol=1e-6, comm=None, verbose=False):
    """
    Solve Ax = b using parallel Gauss-Seidel with hypercube broadcasting.

    Args:
        A: Coefficient matrix (n x n numpy array)
        b: Right-hand side vector (n-element numpy array)
        x0: Initial guess (default: zeros)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        comm: MPI communicator (default: MPI.COMM_WORLD)
        verbose: If True, print iteration information

    Returns:
        x: Solution vector
        iterations: Number of iterations performed
        converged: Whether the method converged
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    n = len(b)

    # Initialize solution vector
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Distribute equations among processes
    # Process i owns equations assigned to it
    equations_per_process = n // size
    remainder = n % size

    # Assign equations to processes (distribute remainder evenly)
    my_equations = []
    eq_idx = 0
    for p in range(size):
        num_eqs = equations_per_process + (1 if p < remainder else 0)
        if p == rank:
            my_equations = list(range(eq_idx, eq_idx + num_eqs))
        eq_idx += num_eqs

    if verbose and rank == 0:
        print(f"Solving {n}x{n} system with {size} processes")
        print(f"Equations per process: {equations_per_process} + {remainder} remainder")

    # Gauss-Seidel iterations
    converged = False
    iteration = 0

    for iteration in range(max_iter):
        x_old = x.copy()

        # Process equations in order from 0 to n-1
        for i in range(n):
            # Determine which process owns equation i
            owner = i // (equations_per_process + 1) if i < remainder * (equations_per_process + 1) else \
                    remainder + (i - remainder * (equations_per_process + 1)) // equations_per_process

            # Owner computes new value for x[i]
            if rank == owner:
                # Compute x_i(t+1) = (b_i - sum(A_ij * x_j for j != i)) / A_ii
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        sigma += A[i, j] * x[j]

                x[i] = (b[i] - sigma) / A[i, i]

            # Broadcast x[i] from owner to all processes using hypercube
            x[i] = hypercube_broadcast(x[i], owner, comm, verbose=False)

        # Check convergence (norm of difference)
        diff_norm = np.linalg.norm(x - x_old, ord=np.inf)

        if verbose and rank == 0 and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: ||x_new - x_old||_inf = {diff_norm:.6e}")

        if diff_norm < tol:
            converged = True
            if verbose and rank == 0:
                print(f"Converged in {iteration + 1} iterations")
            break

    if not converged and verbose and rank == 0:
        print(f"Did not converge after {max_iter} iterations")

    return x, iteration + 1, converged


def create_diagonally_dominant_matrix(n, diagonal_weight=10.0):
    """
    Create a random diagonally dominant matrix (guarantees convergence).

    Args:
        n: Matrix size
        diagonal_weight: How much larger diagonal elements should be

    Returns:
        A: n x n diagonally dominant matrix
    """
    A = np.random.rand(n, n)
    # Make diagonally dominant: A_ii > sum(|A_ij| for j != i)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + diagonal_weight
    return A


def main():
    """Main function when run as standalone script."""
    parser = argparse.ArgumentParser(
        description="Solve linear systems using parallel Gauss-Seidel on hypercube",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="Size of the linear system (n x n)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed iteration information"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Create test problem (only on rank 0, then broadcast)
    if rank == 0:
        np.random.seed(args.seed)
        n = args.size

        # Create diagonally dominant matrix (guarantees convergence)
        A = create_diagonally_dominant_matrix(n)

        # Create exact solution
        x_exact = np.random.rand(n)

        # Compute right-hand side
        b = A @ x_exact

        print(f"Created {n}x{n} diagonally dominant system")
        print(f"Exact solution norm: {np.linalg.norm(x_exact):.6f}")
        print("-" * 60)
    else:
        A = None
        b = None
        x_exact = None

    # Broadcast problem to all processes
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x_exact = comm.bcast(x_exact, root=0)

    # Solve using parallel Gauss-Seidel
    x, iterations, converged = parallel_gauss_seidel(
        A, b,
        max_iter=args.max_iter,
        tol=args.tol,
        comm=comm,
        verbose=args.verbose
    )

    # Report results (from rank 0)
    if rank == 0:
        print("-" * 60)
        if converged:
            print(f"✓ Converged in {iterations} iterations")
        else:
            print(f"✗ Did not converge after {iterations} iterations")

        # Compute error
        error = np.linalg.norm(x - x_exact, ord=np.inf)
        residual = np.linalg.norm(A @ x - b, ord=np.inf)

        print(f"Solution error: ||x - x_exact||_inf = {error:.6e}")
        print(f"Residual: ||Ax - b||_inf = {residual:.6e}")

        if args.verbose:
            print(f"\nComputed solution:\n{x}")
            print(f"\nExact solution:\n{x_exact}")


if __name__ == "__main__":
    main()
