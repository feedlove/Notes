import time
from scipy.special import i0

NUM_REPEAT = 3


def integrate(a, b, N, f):
    """
    Compute the integral of f on the interval [a, b]
    using the mid point rule with N points.
    (https://en.wikipedia.org/wiki/Rectangle_method)
    """
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
    # Parameters
    a = 0.0
    b = 1.0
    N = 100000

    # Benchmark
    t0 = time.time()

    for i in range(NUM_REPEAT):
        # Result should be ~1.0865210970
        res = integrate(a, b, N, f)

    dt = (time.time() - t0) / NUM_REPEAT
    print("\nSerial evaluation ({:.3f} sec): {}\n".format(dt, res))


if __name__ == "__main__":
    main()