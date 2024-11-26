import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Args:
            temperature: Softmax temperature for distillation
            alpha: Weight for balancing distillation and student loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distillation loss between student and teacher
        """
        # Compute soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute soft predictions from student
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Calculate distillation loss (KL divergence)
        distillation_loss = F.kl_div(soft_predictions, soft_targets, 
                                    reduction='batchmean') * (self.temperature ** 2)
        
        # Calculate standard cross entropy loss
        student_loss = F.cross_entropy(student_logits, targets)
        
        # Combine losses
        total_loss = (self.alpha * student_loss + 
                     (1 - self.alpha) * distillation_loss)
        
        return total_loss

class KnowledgeDistillation:
    def __init__(self, 
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 temperature: float = 3.0,
                 alpha: float = 0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = DistillationLoss(temperature, alpha)
        
    def train_step(self, 
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> dict:
        """
        Perform one training step of knowledge distillation
        """
        optimizer.zero_grad()
        
        # Get teacher predictions (no grad needed)
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            
        # Get student predictions
        student_logits = self.student_model(inputs)
        
        # Calculate loss
        loss = self.loss_fn(student_logits, teacher_logits, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracies
        student_acc = self._calculate_accuracy(student_logits, targets)
        teacher_acc = self._calculate_accuracy(teacher_logits, targets)
        
        return {
            'loss': loss.item(),
            'student_acc': student_acc,
            'teacher_acc': teacher_acc
        }
    
    @staticmethod
    def _calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy from logits and targets"""
        predictions = torch.argmax(logits, dim=1)
        return (predictions == targets).float().mean().item()
    
    def distill(self, 
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                device: str = 'cuda',
                callback: Optional[callable] = None) -> list:
        """
        Perform knowledge distillation training
        """
        self.teacher_model.eval()
        self.student_model.train()
        
        history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                'loss': 0.0,
                'student_acc': 0.0,
                'teacher_acc': 0.0
            }
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Perform training step
                step_metrics = self.train_step(inputs, targets, optimizer)
                
                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += step_metrics[key]
                
                if callback:
                    callback(epoch, batch_idx, step_metrics)
            
            # Average epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= len(train_loader)
            
            history.append(epoch_metrics)
            
        return history