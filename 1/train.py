import torch
import torch.nn as nn
import torchvision.transforms as transforms
#import wandb
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange

from example_project.hparams import config
def compute_accuracy(preds, targets):
    return (preds == targets).float().mean()


def create_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                             std=(0.247, 0.243, 0.261)),
    ])


def load_train_dataset(transform):
    return CIFAR10(root='CIFAR10/train',
                   train=True,
                   transform=transform,
                   download=False)


def load_test_dataset(transform):
    return CIFAR10(root='CIFAR10/test',
                   train=False,
                   transform=transform,
                   download=False)


def create_dataloader(dataset, batch_size=None, shuffle=False):
    if batch_size is None:
        batch_size = config["batch_size"]
    return torch.utils.data.DataLoader(dataset=dataset, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle)


def create_model():
    return resnet18(weights=None, 
                   num_classes=10, 
                   zero_init_residual=config["zero_init_residual"])


def create_loss_function():
    return nn.CrossEntropyLoss()


def setup_optimizer(model):
    return torch.optim.AdamW(model.parameters(), 
                            lr=config["learning_rate"], 
                            weight_decay=config["weight_decay"])


def process_batch(batch_idx, batch_data, device, model, criterion, optimizer, test_loader):
    images, labels = batch_data
    images = images.to(device)
    labels = labels.to(device)

    # форвард пасс
    outputs = model(images)
    loss = criterion(outputs, labels)

    # и бэквард пасс + оптимизация
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    metrics = {}
    
    # периодически оцениваем на тесте
    if batch_idx % 100 == 0:
        metrics = evaluate_model(model, test_loader, device, loss)
    
    return metrics


def evaluate_model(model, test_loader, device, train_loss):
    predictions = []
    ground_truth = []

    model.eval()
    with torch.inference_mode():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            
            outputs = model(test_images)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.append(preds)
            ground_truth.append(test_labels)
    
    model.train()
    
    all_preds = torch.cat(predictions)
    all_labels = torch.cat(ground_truth)
    accuracy = compute_accuracy(all_preds, all_labels)
    
    return {'test_acc': accuracy, 'train_loss': train_loss}


def train_model(transform, train_loader, test_loader, model, criterion, optimizer):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    metrics_history = []
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader)):
            metrics = process_batch(batch_idx, batch_data, device, model, 
                                   criterion, optimizer, test_loader)
            
            if batch_idx % 100 == 0:
                metrics_history.append(metrics)
    
    torch.save(model.state_dict(), "model.pt")
    
    return metrics_history

_transform = create_transform
_train_dataset = load_train_dataset
_test_dataset = load_test_dataset
_dataloader = create_dataloader
_model = create_model
_criterion = create_loss_function
_optimizer = setup_optimizer
train_one_batch = process_batch
main = train_model


if __name__ == '__main__':
    transform = create_transform()
    train_dataset = load_train_dataset(transform)
    test_dataset = load_test_dataset(transform)
    
    train_loader = create_dataloader(train_dataset, shuffle=True)
    test_loader = create_dataloader(test_dataset)
    
    model = create_model()
    criterion = create_loss_function()
    optimizer = setup_optimizer(model)
    
    train_model(transform, train_loader, test_loader, model, criterion, optimizer)
