import numpy as np


def f(x, c=1):
	return x ** 2 + c

def gcd(x, y):

	if y > x:
		x, y = y, x

	if x % y == 0:
		print("gcd(%d, %d) = %d" % (x, y, y))
		return y

	print("gcd(%d, %d) = " % (x, y), end="")

	return gcd(y, x % y)



if __name__ == "__main__":
		

	N = 2021
	x = 5
	y = 5

	i = 0

	while True:
		
		x = f(x) % N
		y = f(f(y) % N) % N		

		diff = np.abs(x - y)

		i += 1

		print("x%d = %d; y%d = %d; |x%d - y%d| = %d" % (i, x, i, y, i, i, diff))

		r = gcd(N, diff)

		print("------------------------------------------------------------")

		if r != 1:
			break





