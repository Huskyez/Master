import numpy as np



# array of 4-tuples (x = a * (q) + r)
# 5 tuple (r = q1*a + q2*b)

equations = []

def euclid(x, y):

	if y > x:
		x, y = y, x

	print("%d = %d * (%d) + %d" % (x, x // y, y, x % y))

	equations.append((x % y, 1, x, - (x // y), y))

	if x % y == 1:
		return

	euclid(y, x % y)

	# print("gcd(%d, %d) = " % (x, y), end="")


def substitution():
	
	eq = equations[len(equations)-1]

	print("%d = %d * (%d) + %d * (%d)" % eq)

	for i in range(len(equations) - 2, -1, -1):
		
		b = equations[i]
		a = eq

		print("%d = %d*%d + %d * (%d*%d + %d * %d) <=> " % (a[0], a[1], a[2], a[3], b[1], b[2], b[3], b[4]))

		eq = (a[0], a[3] * b[1], b[2], a[1] + a[3] * b[3], b[4])

		print("%d = %d * (%d) + %d * (%d)" % (eq[0], eq[1], eq[2], eq[3], eq[4]))
	
	return eq

if __name__ == "__main__":
	
	# Compute x ^ (-1) mod N with steps

	x = 11
	N = 42

	x = int(input("x = "))
	N = int(input("N = "))

	euclid(N, x)

	print("--------------------------------------------------------------")

	for eq in equations:
		print("%d = %d * (%d) + %d * (%d)" % eq)

	print("--------------------------------------------------------------")

	eq = substitution()


	print("Final: ")
	print("%d = %d * (%d) + %d * (%d)" % (eq[0], eq[1], eq[2], eq[3] % N, eq[4]))
