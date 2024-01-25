from torchvision import datasets, transforms
from PIL import Image
import os

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
cifar_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a directory to save the images
output_folder = 'cifar_images/train'
os.makedirs(output_folder, exist_ok=True)

# Save images to the output folder
for i in range(len(cifar_dataset)):
    img_tensor, _ = cifar_dataset[i]
    img_array = (img_tensor.numpy() * 255).astype('uint8')  # Convert to numpy array
    img = Image.fromarray(img_array.transpose(1, 2, 0))  # Convert to PIL Image

    # Save as PNG or JPEG
    img.save(os.path.join(output_folder, f'image_{i + 1}.png'))  # Change to 'jpg' for JPEG format

print(f'{len(cifar_dataset)} images saved to {output_folder}.')