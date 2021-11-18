import numpy as np





def euclid(x, y):

	if y > x:
		x, y = y, x

	print("%d = %d * (%d) + %d" % (x, x // y, y, x % y))

	if x % y == 1:
		return

	euclid(y, x % y)

	# print("gcd(%d, %d) = " % (x, y), end="")


if __name__ == "__main__":
	
	# Compute x ^ (-1) mod N with steps

	x = 5
	N = 1932

	euclid(1932, 5)




