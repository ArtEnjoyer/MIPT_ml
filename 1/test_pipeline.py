import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10

from train import (
    compute_accuracy,
    _model,
    _criterion,
    _optimizer,
    _dataloader,
    _test_dataset,
    _train_dataset,
    _transform,
    train_one_batch,
    main
)
TEST_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "epochs": 2,
    "zero_init_residual": False,
}


@pytest.fixture
def data_transform():
    transform = _transform()
    assert transform is not None, "Transform не должен быть None"
    return transform


@pytest.fixture
def training_dataset(data_transform):
    dataset = _train_dataset(data_transform)
    assert dataset is not None, "Наш тренировочный датасет не должен быть None"
    assert len(dataset) > 0, "Наш тренировочный датасет не должен быть пустым"
    return dataset


@pytest.fixture
def validation_dataset(data_transform):
    dataset = _test_dataset(data_transform)
    assert dataset is not None, "Наш валидационный датасет не должен быть None"
    assert len(dataset) > 0, "Наш валидационный датасет не должен быть пустым"
    return dataset


@pytest.fixture
def neural_network():
    model = _model()
    assert isinstance(model, nn.Module), "Модель должна принадлежать классу PyTorch модулей"
    assert hasattr(model, 'fc'), "У модели должен быть fully connected layer!"
    assert model.fc.out_features == 10, "У модели должно быть 10 классов на выходе!!"
    return model


@pytest.mark.parametrize("device", ["cpu"])  # нужно добавить в параметры ["cuda"] если тестируем на GPU
def test_train_on_one_batch(device, training_dataset, validation_dataset, neural_network):
    train_loader = _dataloader(training_dataset, shuffle=True)
    test_loader = _dataloader(validation_dataset)
    
    assert len(train_loader) > 0, "Train dataloader должен быть не пустым"
    assert len(test_loader) > 0, "Test dataloader должен быть не пустым"
    
    model = neural_network
    model.to(device)
    
    criterion = _criterion()
    assert isinstance(criterion, nn.CrossEntropyLoss), "Loss function должна быть CrossEntropyLoss"
    
    optimizer = _optimizer(model)
    assert optimizer is not None, "Optimizer обязан быть не None!!"
    
    batch_data = next(iter(train_loader))
    images, labels = batch_data
    images = images.to(device)
    labels = labels.to(device)
    
    batch_idx = 0  
    metrics = train_one_batch(batch_idx, batch_data, device, model, criterion, optimizer, test_loader)
    
    assert "train_loss" in metrics, "Метрики должны включать training loss"
    assert "test_acc" in metrics, "Метрики должны включать test accuracy"
    assert metrics["test_acc"] >= 0 and metrics["test_acc"] <= 1, "Акураси должна быть между 0 и 1"


def test_model_architecture():
    model = _model()
    
    assert isinstance(model, nn.Module), "Модель должна принадлежать классу PyTorch модулей!"
    assert hasattr(model, 'fc'), "У модели нет fully connected layer!"
    assert model.fc.out_features == 10, "У CIFAR10 размер на выходе должен быть 10!"
    
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN values in {name}"
        assert torch.isfinite(param).all(), f"Infinite values in {name}"


def test_optimizer_configuration():
    model = _model()
    optimizer = _optimizer(model)
    
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == TEST_CONFIG['learning_rate'], "Learning rate должен совпадать с конфигурацией"
        assert param_group['weight_decay'] == TEST_CONFIG['weight_decay'], "Weight decay должен совпадать с конфигурацией"


def test_accuracy_calculation():
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 2, 3])
    assert compute_accuracy(preds, targets) == 1.0, "Все совпало => 100% акураси"
    
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([1, 2, 3, 4])
    assert compute_accuracy(preds, targets) == 0.0, "Ничего не совпало => 0% акураси"
    
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 5, 6])
    assert compute_accuracy(preds, targets) == 0.5, "50% совпало => 50% акураси"


def test_model_save_load(tmp_path):
    model = _model()
    save_path = tmp_path / "test_model.pth"
    
    torch.save(model.state_dict(), save_path)
    assert save_path.exists(), "Модель должна быть сохранена"
    
    loaded_model = _model()
    loaded_model.load_state_dict(torch.load(save_path))
    
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), loaded_model.named_parameters()):
        assert name1 == name2, "Параметры должны совпадать"
        assert torch.allclose(param1, param2), "Значения параметров должны совпадать"


def test_training():
    transform = _transform()
    train_dataset = _train_dataset(transform)
    test_dataset = _test_dataset(transform)
    
    train_loader = _dataloader(train_dataset, shuffle=True)
    test_loader = _dataloader(test_dataset)
    
    model = _model()
    criterion = _criterion()
    optimizer = _optimizer(model)
                              
    metrics_log = main(transform, train_loader, test_loader, model, criterion, optimizer)
    
    assert len(metrics_log) > 0, "Лог должен быть не пустым!!!"
    
    assert metrics_log[-1]["train_loss"] < metrics_log[0]["train_loss"], "Функция потерь должна уменьшаться"
    
    assert metrics_log[-1]["test_acc"] > metrics_log[0]["test_acc"], "Акураси должна расти"
    
    early_improvement = metrics_log[0]["train_loss"] - metrics_log[10]["train_loss"]
    late_improvement = metrics_log[-10]["train_loss"] - metrics_log[-1]["train_loss"]
    assert early_improvement > late_improvement, "Обучение в начале должно проходить быстрее"
    
