import math

def combination_by_index(n, k, index):
    """
    Generate the combination of n elements taken k at a time at a specific index.
    """
    combination = []
    a = n
    b = k
    x = index + 1

    for _ in range(1, k + 1):
        comb = math.comb(a - 1, b)
        while comb >= x:
            a -= 1
            comb = math.comb(a - 1, b)
        combination.append(n - a)
        x -= comb
        a -= 1
        b -= 1

    return combination


def combinations_range(n, k, start, stop):
    """
    Generator that yields combinations of `n` elements taken `k` at a time,
    starting from index `start` to `stop`.
    """
    for idx in range(start, stop):
        yield combination_by_index(n, k, idx)