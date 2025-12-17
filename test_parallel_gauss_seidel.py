#!/usr/bin/env python3
"""
Test suite for parallel Gauss-Seidel implementation.

Run with: mpirun -n <num_processes> python test_parallel_gauss_seidel.py
"""

import numpy as np
import sys
from mpi4py import MPI
from parallel_gauss_seidel import parallel_gauss_seidel, create_diagonally_dominant_matrix


def test_simple_2x2_system(comm, verbose=True):
    """Test a simple 2x2 system with known solution."""
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 1: Simple 2x2 system")
        print("="*60)

    # System: 4x + y = 1, x + 3y = 2
    # Solution: x = 1/11, y = 7/11
    A = np.array([[4.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_exact = np.array([1.0/11.0, 7.0/11.0])

    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-8)

    if rank == 0:
        error = np.linalg.norm(x - x_exact)
        residual = np.linalg.norm(A @ x - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Solution: {x}")
        print(f"Expected: {x_exact}")
        print(f"Error: {error:.2e}")
        print(f"Residual: {residual:.2e}")

        if error < 1e-6 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_3x3_system(comm, verbose=True):
    """Test a 3x3 diagonally dominant system."""
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 2: 3x3 diagonally dominant system")
        print("="*60)

    # Create a diagonally dominant system
    A = np.array([[10.0, 1.0, 2.0],
                  [1.0, 10.0, 1.0],
                  [2.0, 1.0, 10.0]])

    # Known exact solution
    x_exact = np.array([1.0, 2.0, 3.0])
    b = A @ x_exact

    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-8)

    if rank == 0:
        error = np.linalg.norm(x - x_exact)
        residual = np.linalg.norm(A @ x - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Solution: {x}")
        print(f"Expected: {x_exact}")
        print(f"Error: {error:.2e}")
        print(f"Residual: {residual:.2e}")

        if error < 1e-6 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_larger_system(comm, n=10, verbose=True):
    """Test a larger random diagonally dominant system."""
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print(f"TEST 3: {n}x{n} random diagonally dominant system")
        print("="*60)

    if rank == 0:
        np.random.seed(123)
        A = create_diagonally_dominant_matrix(n, diagonal_weight=20.0)
        x_exact = np.random.rand(n)
        b = A @ x_exact
    else:
        A = None
        b = None
        x_exact = None

    # Broadcast to all processes
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x_exact = comm.bcast(x_exact, root=0)

    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-6, max_iter=500)

    if rank == 0:
        error = np.linalg.norm(x - x_exact)
        residual = np.linalg.norm(A @ x - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Error: {error:.2e}")
        print(f"Residual: {residual:.2e}")

        if error < 1e-4 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_identity_system(comm, verbose=True):
    """Test identity matrix (trivial case)."""
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 4: Identity matrix (trivial case)")
        print("="*60)

    n = 5
    A = np.eye(n)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_exact = b.copy()

    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-8)

    if rank == 0:
        error = np.linalg.norm(x - x_exact)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Solution: {x}")
        print(f"Expected: {x_exact}")
        print(f"Error: {error:.2e}")

        if error < 1e-6 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_comparison_with_numpy(comm, verbose=True):
    """Compare results with NumPy's direct solver."""
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 5: Comparison with NumPy direct solver")
        print("="*60)

    if rank == 0:
        np.random.seed(456)
        n = 8
        A = create_diagonally_dominant_matrix(n)
        b = np.random.rand(n)

        # Solve with NumPy
        x_numpy = np.linalg.solve(A, b)
    else:
        A = None
        b = None
        x_numpy = None

    # Broadcast
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x_numpy = comm.bcast(x_numpy, root=0)

    # Solve with parallel Gauss-Seidel
    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-8)

    if rank == 0:
        error = np.linalg.norm(x - x_numpy)
        residual_gs = np.linalg.norm(A @ x - b)
        residual_np = np.linalg.norm(A @ x_numpy - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Difference from NumPy: {error:.2e}")
        print(f"GS Residual: {residual_gs:.2e}")
        print(f"NumPy Residual: {residual_np:.2e}")

        if error < 1e-6 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_large_100x100_system(comm, verbose=True):
    """Test large 100x100 system with timing."""
    import time
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 6: Large 100x100 system")
        print("="*60)

    if rank == 0:
        np.random.seed(789)
        n = 100
        A = create_diagonally_dominant_matrix(n, diagonal_weight=50.0)
        x_exact = np.random.rand(n)
        b = A @ x_exact
    else:
        A = None
        b = None
        x_exact = None

    # Broadcast to all processes
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x_exact = comm.bcast(x_exact, root=0)

    # Time the solution
    start_time = time.time()
    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-6, max_iter=1000)
    elapsed_time = time.time() - start_time

    if rank == 0:
        error = np.linalg.norm(x - x_exact)
        residual = np.linalg.norm(A @ x - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Time: {elapsed_time:.3f} seconds")
        print(f"Error: {error:.2e}")
        print(f"Residual: {residual:.2e}")

        if error < 1e-4 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_large_200x200_system(comm, verbose=True):
    """Test very large 200x200 system with timing."""
    import time
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 7: Very large 200x200 system")
        print("="*60)

    if rank == 0:
        np.random.seed(999)
        n = 200
        A = create_diagonally_dominant_matrix(n, diagonal_weight=100.0)
        x_exact = np.random.rand(n)
        b = A @ x_exact
    else:
        A = None
        b = None
        x_exact = None

    # Broadcast to all processes
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x_exact = comm.bcast(x_exact, root=0)

    # Time the solution
    start_time = time.time()
    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-5, max_iter=2000)
    elapsed_time = time.time() - start_time

    if rank == 0:
        error = np.linalg.norm(x - x_exact)
        residual = np.linalg.norm(A @ x - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Time: {elapsed_time:.3f} seconds")
        print(f"Error: {error:.2e}")
        print(f"Residual: {residual:.2e}")

        if error < 1e-3 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def test_large_500x500_system(comm, verbose=True):
    """Test extra large 500x500 system with timing."""
    import time
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("TEST 8: Extra large 500x500 system")
        print("="*60)

    if rank == 0:
        np.random.seed(1234)
        n = 500
        A = create_diagonally_dominant_matrix(n, diagonal_weight=200.0)
        x_exact = np.random.rand(n)
        b = A @ x_exact
    else:
        A = None
        b = None
        x_exact = None

    # Broadcast to all processes
    A = comm.bcast(A, root=0)
    b = comm.bcast(b, root=0)
    x_exact = comm.bcast(x_exact, root=0)

    # Time the solution
    start_time = time.time()
    x, iters, converged = parallel_gauss_seidel(A, b, comm=comm, verbose=False, tol=1e-5, max_iter=3000)
    elapsed_time = time.time() - start_time

    if rank == 0:
        error = np.linalg.norm(x - x_exact)
        residual = np.linalg.norm(A @ x - b)

        print(f"Converged: {converged}")
        print(f"Iterations: {iters}")
        print(f"Time: {elapsed_time:.3f} seconds")
        print(f"Error: {error:.2e}")
        print(f"Residual: {residual:.2e}")

        if error < 1e-3 and converged:
            print("✓ PASSED")
            return True
        else:
            print("✗ FAILED")
            return False
    return True


def main():
    """Run all tests."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("="*60)
        print("PARALLEL GAUSS-SEIDEL TEST SUITE")
        print(f"Running with {size} MPI processes")
        print("="*60)

    # Run all tests
    results = []
    results.append(test_simple_2x2_system(comm))
    results.append(test_3x3_system(comm))
    results.append(test_larger_system(comm, n=10))
    results.append(test_identity_system(comm))
    results.append(test_comparison_with_numpy(comm))
    results.append(test_large_100x100_system(comm))
    results.append(test_large_200x200_system(comm))
    results.append(test_large_500x500_system(comm))

    # Summary
    if rank == 0:
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")

        if passed == total:
            print("✓ ALL TESTS PASSED")
            sys.exit(0)
        else:
            print("✗ SOME TESTS FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
