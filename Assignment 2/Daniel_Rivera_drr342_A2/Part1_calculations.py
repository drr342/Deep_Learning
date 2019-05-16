import numpy as np

A = np.array([[4,5,2,2,1],[3,3,2,2,4],[4,3,4,1,1],[5,1,4,1,2],[5,1,3,1,4]])
B = np.array([[4,3,3],[5,5,5],[2,4,3]])
C = np.zeros((3,3), dtype=int)

for i in range(3):
	for j in range(3):
		C[i][j] = np.sum(np.multiply(A[np.ix_(range(i,i+3),range(j,j+3))],B))

print(C)

dA = np.zeros((5,5), dtype=int)
for i in range(5):
	for j in range(5):
		for k in range(3):
			for l in range(3):
				if i-k < 0 or i-k > 2 or j-l < 0 or j-l > 2:
					continue
				dA[i][j] += B[i-k][j-l]

print(dA)
