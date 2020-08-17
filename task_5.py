import math

a = (3 + math.sqrt(5)) / 2

def C(n, k):
    if k == n or k == 0:
        return 1
    if k != 1:
        return C(n-1, k) + C(n-1, k-1)
    else:
        return n


def calculate(n):
    return (2013 * (n + 2 * n ** 2)) / ((2 - a) * (1 - n + a * (n ** 2)))

b = 0.1 ** 5 + 0.9 * (0.1 **4) * C(5, 1) + (0.9 ** 2) * (0.1 ** 3) * C(5, 2) + (0.9 ** 3) *  (0.1  ** 2) * C(5, 3) + (0.9 ** 4) * 0.1 * C(5, 4) + 0.9 ** 5
print(b)
