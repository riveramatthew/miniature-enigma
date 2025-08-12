import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a flower classifier")
    parser.add_argument('data_dir', help='Directory of training data')
    parser.add_argument('--save_dir', default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', default='vgg16', help='Pretrained model architecture (vgg16, vgg13, resnet18, etc)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_loader, valid_loader, test_loader, train_dataset

def build_model(arch='vgg16', hidden_units=512, output_size=102):
    # Load pretrained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = model.classifier[0].in_features
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = model.classifier[0].in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = model.fc.in_features
    else:
        raise ValueError(f"Architecture '{arch}' is not supported")

    # Freeze pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # Build classifier
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )

    # Replace classifier / fully connected layer depending on arch
    if arch.startswith('vgg'):
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier

    return model

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.to(device)
    steps = 0
    print_every = 40

    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for val_inputs, val_labels in valid_loader:
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        val_outputs = model(val_inputs)
                        batch_loss = criterion(val_outputs, val_labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(val_outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == val_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

def save_checkpoint(model, train_dataset, save_dir, arch, hidden_units, learning_rate, epochs):
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, train_dataset = load_data(args.data_dir)

    model = build_model(args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    if args.arch.startswith('resnet'):
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)

    save_checkpoint(model, train_dataset, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == '__main__':
    main()
