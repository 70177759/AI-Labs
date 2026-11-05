import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ---------- 1. Input Image (from your notebook) ----------
image = np.array([
    [8, 0, 3, 5, 6],
    [6, 1, 0, 6, 8],
    [1, 5, 2, 2, 0],
    [5, 4, 1, 1, 1],
    [0, 3, 0, 7, 4],
    [1, 1, 1, 9, 3],
], dtype=float)

kernel = np.array([
    [ 1, 3,  1, 1],
    [ 0, 4,  3, 0],
    [-1, 2, -9, 1],
    [ 3, 4,  6, 0],
], dtype=float)


# ---------- 2. Convolution + ReLU ----------
def convolve2d(img, k, stride=1):
    """'Valid' convolution (cross-correlation, as used in CNNs)."""
    kh, kw = k.shape
    out_h = (img.shape[0] - kh) // stride + 1
    out_w = (img.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(patch * k)
    return out


def relu(x):
    return np.maximum(0, x)


# ---------- 3. Pooling ----------
def max_pool(x, size=2, stride=1):
    out_h = (x.shape[0] - size) // stride + 1
    out_w = (x.shape[1] - size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = x[i*stride:i*stride+size, j*stride:j*stride+size].max()
    return out


# ---------- 4. Flatten ----------
def flatten(x):
    return x.reshape(-1)


# ---------- 5. Fully Connected ----------
def fully_connected(x, W, b):
    return W @ x + b


# ---------- 6. Softmax ----------
def softmax(z):
    e = np.exp(z - np.max(z))   # subtract max for numerical stability
    return e / e.sum()


# ---------- Run the pipeline ----------
if __name__ == "__main__":
    print("1. Input image", image.shape, "\n", image, "\n")

    conv_out = convolve2d(image, kernel)
    print("2a. Convolution", conv_out.shape, "\n", conv_out, "\n")

    relu_out = relu(conv_out)
    print("2b. After ReLU\n", relu_out, "\n")

    pooled = max_pool(relu_out, size=2, stride=1)
    print("3. Max pooling (2x2, stride 1)", pooled.shape, "\n", pooled, "\n")

    flat = flatten(pooled)
    print("4. Flatten", flat.shape, "\n", flat, "\n")

    # Fully connected layer: 3 output classes (weights are random but fixed by seed)
    rng = np.random.default_rng(42)
    num_classes = 3
    W = rng.normal(0, 0.1, size=(num_classes, flat.size))
    b = np.zeros(num_classes)

    # scale down so softmax is not saturated by large conv values
    scores = fully_connected(flat / flat.max() if flat.max() > 0 else flat, W, b)
    print("5. Fully connected scores\n", scores, "\n")

    probs = softmax(scores)
    print("6. Softmax probabilities\n", probs, "  (sum =", probs.sum().round(4), ")")
    print("Predicted class:", int(np.argmax(probs)))