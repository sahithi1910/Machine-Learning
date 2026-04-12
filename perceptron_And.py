# Simple step function
def step(x):
    return 1 if x >= 0 else 0

# Training data (AND gate)
data = [(0,0), (0,1), (1,0), (1,1)]
targets = [0, 0, 0, 1]

# Initialize weights (bias, w1, w2)
w0, w1, w2 = 0, 0, 0
lr = 0.1

# Training
for epoch in range(10):
    for i in range(len(data)):
        x1, x2 = data[i]

        # Calculate output
        y = w0*1 + w1*x1 + w2*x2
        output = step(y)

        # Error
        error = targets[i] - output

        # Update weights
        w0 += lr * error * 1
        w1 += lr * error * x1
        w2 += lr * error * x2

# Testing
print("Testing AND gate:")
for x1, x2 in data:
    y = w0*1 + w1*x1 + w2*x2
    print(x1, x2, "→", step(y))