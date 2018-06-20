import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value
alpha = 0.01

def forward(x):
    return x * w

# loss function
def loss(x, y):
    y_pred = forward(x)
    return np.square((y_pred - y))

# compute gradient
def gradient(x, y):
    return 2 * x * (x * w - y)

# before training
print("predict (before training)", "4 hours", forward(4))

# training process
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad  = gradient(x_val, y_val)
        w = w - alpha * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)
    print("progress: ", epoch, "w = ", round(w, 2), "loss = ", round(l, 2))

# after training
print("predict (after training)", "4 hours", forward(4))
