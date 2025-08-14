# predict.py - Optimized for High Performance Inference
import torch
from torch import nn
from torch.cuda.amp import autocast
from torchvision import models, transforms
from PIL import Image
import argparse
import json
import time
import os
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

class OptimizedPredictor:
    """High-performance inference class with caching and optimizations."""
    
    def __init__(self, checkpoint_path, device, use_amp=False, compile_model=False):
        self.device = device
        self.use_amp = use_amp
        self.model = None
        self.idx_to_class = None
        self.transform = None
        self._setup_transform()
        self._load_and_optimize_model(checkpoint_path, compile_model)
    
    def _setup_transform(self):
        """Pre-compile transform pipeline for faster image processing."""
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_and_optimize_model(self, checkpoint_path, compile_model):
        """Load model with optimizations for inference."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Load checkpoint with memory mapping for large files
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        arch = checkpoint['arch']
        hidden_units = checkpoint['hidden_units']
        
        # Build model architecture
        if arch == "resnet18":
            self.model = models.resnet18(weights=None)  # Don't load pretrained weights
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, hidden_units),
                nn.ReLU(inplace=True),  # In-place for memory efficiency
                nn.Dropout(0.4),
                nn.Linear(hidden_units, 102),
                nn.LogSoftmax(dim=1)
            )
        elif arch == "vgg13":
            self.model = models.vgg13(weights=None)
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(hidden_units, 102),
                nn.LogSoftmax(dim=1)
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        
        # Create reverse mapping for fast lookup
        self.idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        
        # Move to device and optimize for inference
        self.model.to(self.device)
        self.model.eval()
        
        # Disable gradient computation for inference
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Compile model for PyTorch 2.0+ optimization
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model for optimized inference...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Optimize for inference
        if self.device.type == 'cuda':
            self.model.half() if self.use_amp else self.model.float()
        
        print(f"Model loaded and optimized for {self.device}")
    
    @lru_cache(maxsize=128)
    def _load_and_process_image_cached(self, image_path, image_mtime):
        """Cache processed images to avoid reprocessing the same image."""
        return self._load_and_process_image(image_path)
    
    def _load_and_process_image(self, image_path):
        """Optimized image loading and preprocessing."""
        try:
            # Fast image loading with optimized settings
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply transforms
                tensor = self.transform(img)
                
                # Move to device and add batch dimension
                if self.device.type == 'cuda':
                    tensor = tensor.to(self.device, non_blocking=True)
                    if self.use_amp:
                        tensor = tensor.half()
                else:
                    tensor = tensor.to(self.device)
                
                return tensor.unsqueeze(0)
        
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {str(e)}")
    
    def predict_single(self, image_path, top_k=5):
        """High-performance single image prediction."""
        # Use cached preprocessing if available
        try:
            image_mtime = os.path.getmtime(image_path)
            image_tensor = self._load_and_process_image_cached(image_path, image_mtime)
        except:
            image_tensor = self._load_and_process_image(image_path)
        
        # Inference with optimizations
        with torch.no_grad():
            if self.use_amp and self.device.type == 'cuda':
                with autocast():
                    output = self.model(image_tensor)
            else:
                output = self.model(image_tensor)
            
            # Fast top-k computation
            ps = torch.exp(output)
            top_probs, top_indices = ps.topk(top_k, dim=1, largest=True, sorted=True)
        
        # Convert to numpy and get class names
        probs = top_probs.cpu().float().numpy()[0]
        classes = [self.idx_to_class[idx.item()] for idx in top_indices[0]]
        
        return probs, classes
    
    def predict_batch(self, image_paths, top_k=5):
        """Batch prediction for multiple images (more efficient)."""
        if len(image_paths) == 1:
            return [self.predict_single(image_paths[0], top_k)]
        
        # Load and preprocess all images
        batch_tensors = []
        for img_path in image_paths:
            try:
                image_mtime = os.path.getmtime(img_path)
                tensor = self._load_and_process_image_cached(img_path, image_mtime)
                batch_tensors.append(tensor.squeeze(0))  # Remove batch dim
            except:
                tensor = self._load_and_process_image(img_path)
                batch_tensors.append(tensor.squeeze(0))
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors)
        
        # Batch inference
        with torch.no_grad():
            if self.use_amp and self.device.type == 'cuda':
                with autocast():
                    outputs = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)
            
            # Fast top-k for entire batch
            ps = torch.exp(outputs)
            top_probs, top_indices = ps.topk(top_k, dim=1, largest=True, sorted=True)
        
        # Process results
        results = []
        probs_np = top_probs.cpu().float().numpy()
        indices_np = top_indices.cpu().numpy()
        
        for i in range(len(image_paths)):
            probs = probs_np[i]
            classes = [self.idx_to_class[idx] for idx in indices_np[i]]
            results.append((probs, classes))
        
        return results

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained model.")
    parser.add_argument("input", help="Path to input image or directory of images")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K predictions")
    parser.add_argument("--category_names", help="Path to JSON file mapping categories to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision for faster inference")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for multiple images")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark timing")
    return parser.parse_args()

@lru_cache(maxsize=1)
def load_category_names(category_names_path):
    """Cache category names to avoid repeated file I/O."""
    with open(category_names_path, 'r') as f:
        return json.load(f)

def format_predictions(probs, classes, category_names_path=None):
    """Format prediction results with optional category name mapping."""
    if category_names_path:
        cat_to_name = load_category_names(category_names_path)
        class_names = [cat_to_name.get(c, c) for c in classes]
    else:
        class_names = classes
    
    results = []
    for prob, name in zip(probs, class_names):
        results.append(f"{name}: {prob*100:.2f}%")
    
    return results

def get_image_files(input_path):
    """Get list of image files from input path (file or directory)."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for file in os.listdir(input_path):
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(input_path, file))
        return sorted(image_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

def main():
    args = get_input_args()
    
    # Setup device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize optimized predictor
    start_time = time.time()
    predictor = OptimizedPredictor(
        args.checkpoint, 
        device, 
        use_amp=args.mixed_precision,
        compile_model=args.compile
    )
    init_time = time.time() - start_time
    print(f"Model initialization time: {init_time:.3f}s")
    
    # Get image files
    image_files = get_image_files(args.input)
    print(f"Processing {len(image_files)} image(s)")
    
    # Benchmark mode
    if args.benchmark:
        print("\n=== BENCHMARK MODE ===")
        # Warmup
        if image_files:
            for _ in range(3):
                predictor.predict_single(image_files[0], args.top_k)
        
        # Time single prediction
        if image_files:
            start = time.time()
            for _ in range(10):
                predictor.predict_single(image_files[0], args.top_k)
            single_time = (time.time() - start) / 10
            print(f"Average single prediction time: {single_time*1000:.1f}ms")
        
        # Time batch prediction if multiple images
        if len(image_files) > 1:
            batch_files = image_files[:min(args.batch_size, len(image_files))]
            start = time.time()
            predictor.predict_batch(batch_files, args.top_k)
            batch_time = time.time() - start
            print(f"Batch prediction time ({len(batch_files)} images): {batch_time:.3f}s")
            print(f"Average per image in batch: {batch_time/len(batch_files)*1000:.1f}ms")
    
    # Run predictions
    start_time = time.time()
    
    if len(image_files) == 1:
        # Single image prediction
        probs, classes = predictor.predict_single(image_files[0], args.top_k)
        results = format_predictions(probs, classes, args.category_names)
        
        print(f"\nPredictions for {image_files[0]}:")
        for result in results:
            print(f"  {result}")
    
    else:
        # Batch processing for multiple images
        batch_size = args.batch_size
        all_results = []
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_results = predictor.predict_batch(batch_files, args.top_k)
            
            for img_path, (probs, classes) in zip(batch_files, batch_results):
                results = format_predictions(probs, classes, args.category_names)
                all_results.append((img_path, results))
        
        # Display results
        for img_path, results in all_results:
            print(f"\nPredictions for {os.path.basename(img_path)}:")
            for result in results:
                print(f"  {result}")
    
    total_time = time.time() - start_time
    avg_time = total_time / len(image_files)
    print(f"\nTotal inference time: {total_time:.3f}s")
    print(f"Average per image: {avg_time*1000:.1f}ms")
    print(f"Throughput: {len(image_files)/total_time:.1f} images/sec")

if __name__ == "__main__":
    main()