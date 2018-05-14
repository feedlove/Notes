import multiprocessing as mp
import functools as ft
import time
import sys
from scipy.special import i0

NUM_REPEAT = 3


def integrate(I, N, f):
    """
    Compute the integral of f on the interval I = [a, b]
    using the mid point rule with N points.
    (https://en.wikipedia.org/wiki/Rectangle_method)
    """
    a, b = I
    h = float(b - a)/N   # Step size
    mid_points = (a + h/2 + i*h for i in range(N))

    int_sum = 0.0
    for p in mid_points:
        int_sum += f(p)

    integral = h * int_sum
    return integral


def f(x):
    # Modified Bessel function of order 0
    return i0(x)


def main():
    num_proc = 2

    # Read number of processes from command line
    if len(sys.argv) > 1:
        try:
            num_proc = int(sys.argv[1])
        except ValueError:
            print("Error: Could not parse argument")
            sys.exit(1)

    # Starting subprocesses
    pool = mp.Pool(num_proc)

    # Parameters
    a = 0.0
    b = 1.0
    N = 100000

    # Subproblems
    sub_len = float(b - a)/num_proc
    sub_intervals = [(a + i*sub_len, a + (i+1)*sub_len) for i in range(num_proc)]
    sub_N = N // num_proc
    partial_func = ft.partial(integrate, N=sub_N, f=f)

    # Benchmark
    t0 = time.time()

    for i in range(NUM_REPEAT):
        # Result should be ~1.0865210970
        res = sum(pool.map(partial_func, sub_intervals))

    dt = (time.time() - t0) / NUM_REPEAT
    print("\nParallel evaluation ({:.3f} sec, num_proc={}): {}\n".format(dt, num_proc, res))


if __name__ == "__main__":
    main()