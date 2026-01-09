def input_square_matrix(n):
    matrix = []
    for i in range(n):
        row = list(map(int, input(f"Enter elements of row {i+1}: ").split())) #We Take row as the input and split whereever there is space
        matrix.append(row) #append the rows into the matrix
    return matrix
def multiply_matrix(A, B):
    n = len(A)
    result = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]  #MAtrix Multiplication
    return result
def matrix_power(A, m):
    result = A
    for _ in range(1, m):
        result = multiply_matrix(result, A)  #we call matrix multiplication function m times
    return result
n = int(input("Enter the order of the square matrix: "))
A = input_square_matrix(n)
m = int(input("Enter the power m: "))
result = matrix_power(A, m)   # We get A^m
print("\nMatrix A^", m, "is:")
for row in result:
    print(row)
