# train.py - Optimized for Performance
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
from torchvision import datasets, transforms, models
import argparse
import os
from torch.utils.data import DataLoader
import time

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint")
    parser.add_argument("data_dir", help="Dataset directory")
    parser.add_argument("--save_dir", default=".", help="Directory to save checkpoints")
    parser.add_argument("--arch", default="resnet18", help="Model architecture (resnet18, vgg13, etc.)")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_units", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (increased from 32)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")
    return parser.parse_args()

def load_data(data_dir, batch_size, num_workers):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Optimized transforms with minimal overhead
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Optimized DataLoaders
    trainloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Prefetch batches
    )
    
    validloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return trainloader, validloader, train_dataset

def build_model(arch, hidden_units, learning_rate, compile_model=False):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-6
        )
        
    elif arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        for param in model.features.parameters():
            param.requires_grad = False
        
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.AdamW(
            model.classifier.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-6
        )
    else:
        raise ValueError("Unsupported architecture")

    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, mode="max-autotune")

    criterion = nn.NLLLoss()
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, scheduler, trainloader, validloader, device, epochs, use_amp=False):
    model.to(device)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Calculate total steps for scheduler
    total_steps = len(trainloader) * epochs
    if hasattr(scheduler, 'total_steps'):
        scheduler.total_steps = total_steps
    
    print(f"Training on {device} with {'mixed precision' if use_amp else 'full precision'}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if use_amp:
                with autocast():
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            running_loss += loss.item()
            batch_count += 1
        
        # Validation phase - only every epoch
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        logps = model(inputs)
                        loss = criterion(logps, labels)
                else:
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                
                valid_loss += loss.item()
                
                # More efficient accuracy calculation
                _, predicted = torch.max(logps.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Print statistics
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / len(trainloader)
        val_loss = valid_loss / len(validloader)
        accuracy = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")

def save_checkpoint(model, train_dataset, save_dir, arch, hidden_units, learning_rate, epochs):
    # Extract base model if compiled
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model_to_save.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'learning_rate': learning_rate,
        'epochs': epochs
    }
    
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

def main():
    args = get_input_args()
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print("Using CPU")

    # Load data
    trainloader, validloader, train_dataset = load_data(
        args.data_dir, args.batch_size, args.num_workers
    )
    
    # Build model
    model, criterion, optimizer = build_model(
        args.arch, args.hidden_units, args.learning_rate, args.compile
    )

    # OneCycleLR scheduler now that trainloader and epochs are known
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,
        epochs=args.epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3
    )

    if torch.cuda.device_count() > 1 and args.gpu:
        model = DataParallel(model)
    
    start_time = time.time()
    train_model(
        model, criterion, optimizer, scheduler, 
        trainloader, validloader, device, args.epochs, 
        args.mixed_precision
    )
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f} seconds")
    
    save_checkpoint(
        model, train_dataset, args.save_dir, args.arch, 
        args.hidden_units, args.learning_rate, args.epochs
    )

if __name__ == "__main__":
    main()