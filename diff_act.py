import matplotlib.pyplot as plt
import math

# --------------------------
# Activation Functions
# --------------------------
def bipolar(y):
    return 1 if y >= 0 else -1

def sigmoid(y):
    return 1 / (1 + math.exp(-y))

def relu(y):
    return max(0, y)

# --------------------------
# Data (AND Gate)
# --------------------------
data = [(0,0), (0,1), (1,0), (1,1)]
targets = [0, 0, 0, 1]

# --------------------------
# Training Function
# --------------------------
def train(activation, name, lr=0.05, max_epochs=1000):
    # Initial weights (given)
    w0, w1, w2 = 10, 0.2, -0.75
    
    errors = []
    epochs_taken = 0

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(data)):
            x1, x2 = data[i]

            # Weighted sum
            y = w0 + w1*x1 + w2*x2

            # Output conversion (important!)
            if activation == sigmoid:
                out = 1 if activation(y) >= 0.5 else 0
            elif activation == relu:
                out = 1 if activation(y) > 0 else 0
            elif activation == bipolar:
                out = 1 if activation(y) == 1 else 0

            # Error
            error = targets[i] - out
            total_error += error**2

            # Weight update (Perceptron rule)
            w0 += lr * error
            w1 += lr * error * x1
            w2 += lr * error * x2

        errors.append(total_error)
        epochs_taken = epoch + 1

        # Convergence condition
        if total_error <= 0.002:
            print(f"{name} converged at epoch: {epochs_taken}")
            break

    return errors, epochs_taken

# --------------------------
# A3: Train for all activations
# --------------------------
err_bipolar, ep_bipolar = train(bipolar, "Bipolar")
err_sigmoid, ep_sigmoid = train(sigmoid, "Sigmoid")
err_relu, ep_relu = train(relu, "ReLU")

# --------------------------
# Plot 1: Separate graphs
# --------------------------
plt.plot(err_bipolar)
plt.title("Bipolar Activation")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

plt.plot(err_sigmoid)
plt.title("Sigmoid Activation")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

plt.plot(err_relu)
plt.title("ReLU Activation")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# --------------------------
# Plot 2: Comparison graph
# --------------------------
plt.plot(err_bipolar, label="Bipolar")
plt.plot(err_sigmoid, label="Sigmoid")
plt.plot(err_relu, label="ReLU")

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Activation Function Comparison")
plt.legend()
plt.show()

# --------------------------
# A4: Learning Rate Analysis
# --------------------------
lr_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
epochs_list = []

for lr in lr_values:
    _, epochs = train(bipolar, f"Bipolar (lr={lr})", lr=lr)
    epochs_list.append(epochs)

# Plot Learning Rate vs Epochs
plt.plot(lr_values, epochs_list, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("Learning Rate vs Convergence")
plt.show()