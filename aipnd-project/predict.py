import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json
import os

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    model = build_model(arch, hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def build_model(arch='vgg16', hidden_units=512, output_size=102):
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

    for param in model.parameters():
        param.requires_grad = False

    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_units, output_size),
        torch.nn.LogSoftmax(dim=1)
    )

    if arch.startswith('vgg'):
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier

    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    
    # Resize so shortest side = 256
    width, height = pil_image.size
    if width < height:
        new_width = 256
        new_height = int(height * 256 / width)
    else:
        new_height = 256
        new_width = int(width * 256 / height)
    pil_image = pil_image.resize((new_width, new_height))
    
    # Center crop 224x224
    left = (new_width - 224)/2
    top = (new_height - 224)/2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert to numpy and normalize
    np_image = np.array(pil_image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose color channel to first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, device, topk=5):
    model.to(device)
    model.eval()

    np_image = process_image(image_path)
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.forward(tensor_image)
    ps = torch.exp(output)
    top_p, top_class_idx = ps.topk(topk)

    top_p = top_p.cpu().numpy().squeeze()
    top_class_idx = top_class_idx.cpu().numpy().squeeze()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_class_idx]

    return top_p, top_classes

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, device, args.top_k)

    # Load category names
    if os.path.isfile(args.category_names):
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        class_names = classes

    print("Top probabilities:", probs)
    print("Top classes:", class_names)

if __name__ == '__main__':
    main()
