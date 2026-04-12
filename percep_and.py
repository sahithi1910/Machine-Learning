import matplotlib.pyplot as plt

# Step activation
def bipolar(y):
    return 1 if y >= 0 else 0

# Data (AND gate)
data = [(0,0), (0,1), (1,0), (1,1)]
targets = [0, 0, 0, 1]

# Given initial weights
w0, w1, w2 = 10, 0.2, -0.75
lr = 0.05

errors_per_epoch = []
max_epochs = 1000

for epoch in range(max_epochs):
    total_error = 0

    for i in range(len(data)):
        x1, x2 = data[i]

        # Output
        y = w0 + w1*x1 + w2*x2
        output = step(y)

        # Error
        error = targets[i] - output
        total_error += error**2

        # Update weights
        w0 += lr * error
        w1 += lr * error * x1
        w2 += lr * error * x2

    errors_per_epoch.append(total_error)

    # Convergence condition
    if total_error <= 0.002:
        print("Converged at epoch:", epoch+1)
        break

# Plot
plt.plot(errors_per_epoch)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Error vs Epochs")
plt.show()