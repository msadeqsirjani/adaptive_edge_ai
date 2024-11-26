import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
import time
from pathlib import Path
from tqdm import tqdm

from src.models.dynamic_nn import DynamicNeuralNetwork
from src.optimization.knowledge_distillation import KnowledgeDistillation
from src.utils.metrics import evaluate_model
from src.optimization.compression import ModelCompression

def create_data_loaders(batch_size=64, num_workers=4):
    """Create train and validation data loaders"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def train_teacher_model(model, train_loader, val_loader, device, epochs=10):
    """Train the teacher model"""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Log metrics
        logging.info(f'Epoch {epoch+1}/{epochs}:')
        logging.info(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_teacher_model.pth')
            logging.info('Saved best model checkpoint')

def main():
    # Setup
    setup_logging()
    logging.info("Starting training pipeline")
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        batch_size=64,
        num_workers=4
    )
    logging.info("Data loaders created successfully")
    
    # Create and train teacher model
    teacher_model = DynamicNeuralNetwork(input_channels=3, num_classes=10)
    teacher_model = teacher_model.to(device)
    logging.info("Training teacher model...")
    train_teacher_model(teacher_model, train_loader, val_loader, device)
    
    # Create and initialize student model
    student_model = DynamicNeuralNetwork(input_channels=3, num_classes=10)
    student_model.adjust_complexity(0.5)  # Make student smaller
    student_model = student_model.to(device)
    logging.info("Student model created with 0.5 complexity")
    
    # Initialize knowledge distillation
    distillation = KnowledgeDistillation(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=3.0,
        alpha=0.5
    )
    
    # Initialize optimizer for student model
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    # Perform knowledge distillation
    logging.info("Starting knowledge distillation...")
    def distillation_callback(epoch, batch_idx, metrics):
        if batch_idx % 100 == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {metrics['loss']:.4f}")
    
    history = distillation.distill(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        callback=distillation_callback
    )
    
    # Evaluate the distilled student model
    logging.info("Evaluating student model...")
    student_metrics = evaluate_model(
        model=student_model,
        dataloader=val_loader,
        device=device
    )
    
    logging.info("\nStudent Model Metrics:")
    for metric_name, value in student_metrics.items():
        if isinstance(value, (float, int)):
            logging.info(f"{metric_name}: {value:.4f}")
    
    # Apply compression to the student model
    logging.info("Applying model compression...")
    compressor = ModelCompression(student_model)
    compressor.apply_pruning(amount=0.3)
    
    # Evaluate again after compression
    logging.info("Evaluating compressed model...")
    compressed_metrics = evaluate_model(
        model=student_model,
        dataloader=val_loader,
        device=device
    )
    
    logging.info("\nCompressed Student Model Metrics:")
    for metric_name, value in compressed_metrics.items():
        if isinstance(value, (float, int)):
            logging.info(f"{metric_name}: {value:.4f}")
    
    # Save final models
    torch.save(student_model.state_dict(), 'outputs/final_student_model.pth')
    
    # Export for edge deployment
    logging.info("Exporting model for edge deployment...")
    sample_input = next(iter(val_loader))[0][:1].to(device)
    compressor.export_onnx(sample_input, 'outputs/edge_model.onnx')
    
    logging.info("Training pipeline completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)