
# **Exercise 1**: Creating a Grayscale Image

### Generating and Converting an Image

```python
import torch
import numpy as np
from PIL import Image

# Generate a random 256x256 RGB image
rand_img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
img = Image.fromarray(rand_img)
img.show()

# Convert the image to a tensor
img_np = np.array(img).astype(np.float32)  # Convert to float32 for PyTorch
img_tensor = torch.tensor(img_np)

print('Tensor shape:', img_tensor.shape)

# Calculate the naive grayscale image by averaging the channels
gray_naive = torch.mean(img_tensor, dim=2)
print('Naive Grayscale Tensor shape:', gray_naive.shape)

# Weighted grayscale conversion
weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
weighted_channels = img_tensor @ weights
print('Weighted Grayscale Tensor shape:', weighted_channels.shape)
```

### Moving Tensors to GPU and Recalculating

```python
# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move tensors to GPU
img_gpu = img_tensor.to(device)
weights_gpu = weights.to(device)
gray_gpu = torch.mean(img_gpu, dim=2)
weighted_channels_gpu = (img_gpu @ weights_gpu).to(device)

print(f"Image is on device {img_gpu.device}!")
print(f"Weighted grayscale is on device {weighted_channels_gpu.device}!")

# Move result back to CPU
gray_cpu = gray_gpu.to('cpu')
print(f"Weighted grayscale on CPU: {gray_cpu.device}")
```

## **Exercise 2**: Calculate the Partial Derivatives

```python
import torch

# Initialize tensors on GPU with requires_grad=True for w and b
w = torch.tensor([0.2, 0.5], dtype=torch.float32, requires_grad=True, device='cuda')
b = torch.tensor([0.1, 0.7], dtype=torch.float32, requires_grad=True, device='cuda')

# Initialize tensors on GPU for x and y without requires_grad
x = torch.tensor([2, 3], dtype=torch.float32, device='cuda')
y = torch.tensor([5, 7], dtype=torch.float32, device='cuda')

# Forward step
y_hat = w * x + b
L = torch.sum(0.5 * (y - y_hat) ** 2)

# Backward step
L.backward()

# Print the gradients
print(f"Partial derivative with respect to w: {w.grad}")
print(f"Partial derivative with respect to b: {b.grad}")

# Print the gradient of x (which should be None)
print(f"Partial derivative with respect to x: {x.grad}")  # None because x does not require grad
```

### Alternative Calculation of L

```python
# Compute the loss
L = 0.5 * (y - y_hat) ** 2

# Sum to get a scalar
L_sum = torch.sum(L)

# Backward step
L_sum.backward()

# Print the gradients
print(f"Partial derivative with respect to w: {w.grad}")
print(f"Partial derivative with respect to b: {b.grad}")
print(f"Partial derivative with respect to x: {x.grad}")  # None, as before
```

## **Exercise 3**: Freezing Network Weights

```python
import torch
import torchvision

# Load the pretrained ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Feed a random image through the model
img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
y_hat = model(img)

# Arbitrary label
y = torch.zeros_like(y_hat)
y[0, 3] = 1

# Calculate Euclidean distance (L2 norm)
loss = torch.nn.functional.mse_loss(y_hat, y).sqrt()

# Call backward
loss.backward()  # This will not work because gradients are frozen

# Unfreeze the weights
for param in model.parameters():
    param.requires_grad = True

# Calculate again after unfreezing
y_hat_unfrozen = model(img)
loss_unfrozen = torch.nn.functional.mse_loss(y_hat_unfrozen, y).sqrt()

print(f'Euclidean Distance (L2 norm) after unfreezing: {loss_unfrozen.item()}')
```

In summary:

- **Exercise 1**: Creates a grayscale image from an RGB image and demonstrates tensor operations on both CPU and GPU.
- **Exercise 2**: Calculates and prints the partial derivatives of a loss function with respect to weights and biases.
- **Exercise 3**: Demonstrates how to freeze and unfreeze model parameters and the effect of this on gradient computation.
