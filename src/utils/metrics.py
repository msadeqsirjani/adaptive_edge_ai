import torch
import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.inference_times = []
        self.batch_sizes = []
        
    def update(self, 
               predictions: torch.Tensor,
               targets: torch.Tensor,
               loss: float,
               inference_time: float,
               batch_size: int):
        """Update metrics with batch results"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
        self.inference_times.append(inference_time)
        self.batch_sizes.append(batch_size)
        
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = np.mean(predictions == targets)
        avg_loss = np.mean(self.losses)
        
        # Precision, recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, predictions)
        
        # Inference performance metrics
        avg_inference_time = np.mean(self.inference_times)
        throughput = np.sum(self.batch_sizes) / np.sum(self.inference_times)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'avg_inference_time': avg_inference_time,
            'throughput': throughput
        }

class EdgeMetrics:
    """Specific metrics for edge deployment"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.memory_usage = []
        self.power_consumption = []  # If available
        self.latency = []
        
    def measure_inference_metrics(self, 
                                model: torch.nn.Module,
                                input_tensor: torch.Tensor,
                                num_runs: int = 100) -> Dict[str, float]:
        """Measure inference metrics on edge device"""
        latencies = []
        memory_usage = []
        
        # Warmup runs
        for _ in range(5):
            _ = model(input_tensor)
            
        # Actual measurements
        for _ in range(num_runs):
            torch.cuda.empty_cache()  # If using GPU
            
            # Measure memory before
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Measure inference time
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            latencies.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
        
        return {
            'mean_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'memory_usage': np.mean(memory_usage),
            'latency_std': np.std(latencies)
        }

def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: str = 'cuda') -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate model performance and compute all metrics
    """
    model.eval()
    metrics = MetricsTracker()
    edge_metrics = EdgeMetrics()
    
    total_inference_time = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.perf_counter()
            outputs = model(inputs)
            inference_time = time.perf_counter() - start_time
            total_inference_time += inference_time
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate final metrics
    accuracy = 100. * correct / total
    avg_inference_time = total_inference_time / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time,
        'total_samples': total
    }