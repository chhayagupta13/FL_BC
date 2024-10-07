import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the trained federated model
federated_model = torch.load("/content/drive/MyDrive/vgg_approach1.keras", map_location=torch.device('cpu'))
federated_model.eval()

# Move the model to the same device as the input data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
federated_model.to(device)

# Define the data transforms for the evaluation dataset
eval_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the evaluation dataset
eval_data_dir = '/content/drive/MyDrive/masterdatacleaned'
eval_dataset = datasets.ImageFolder(eval_data_dir, eval_data_transforms)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(eval_dataloader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = federated_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the evaluation dataset: %d %%' % (100 * correct / total))
