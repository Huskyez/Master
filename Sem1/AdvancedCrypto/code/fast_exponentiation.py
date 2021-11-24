import numpy as np



if __name__ == "__main__":
	

	x = 30
	e = 23
	N = 43

	# Compute x^e mod N


	x = int(input("x = "))
	e = int(input("e = "))
	N = int(input("N = "))

	results = []

	bin_e = bin(e)[2:]

	print("%d = " % (e), end="")
	
	for i in range(len(bin_e)):
		if bin_e[i] == "1":
			print("%d + " % (pow(2, len(bin_e) - i - 1)), end="")

	print()
	bin_e = bin_e[::-1]
	# print(bin_e)
	

	i = 1
	init_x = x

	print("%d = %d mod %d" % (init_x, init_x, N))

	if bin_e[0] == "1":
		results.append(init_x)

	for c in bin_e[1:]:

		print("%d^%d = (%d)^2 = %d = %d mod %d" % (init_x, pow(2, i), x, x * x, (x * x) % N, N))
		
		x *= x
		x %= N

		if N - x <= 10:
			x = x - N

		if c == "1":
			results.append(x)
		
		i += 1


	print("%d^%d mod %d = " % (init_x, e, N), end="")
	for y in results:
		print(y, end=" * ")
	print(" mod %d = %d" % (N, np.prod(results) % N))




