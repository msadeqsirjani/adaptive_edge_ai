import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ModelCompression:
    def __init__(self, model: nn.Module):
        self.model = model
        
    def apply_pruning(self, amount=0.3):
        """Apply structured pruning to the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=amount)
                
    def quantize_model(self, dtype=torch.qint8):
        """Quantize the model to reduce memory footprint"""
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        # Note: Model needs to be calibrated with data before conversion
        torch.quantization.convert(self.model, inplace=True)
        
    def export_onnx(self, sample_input, path='model.onnx'):
        """Export model to ONNX format for edge deployment"""
        torch.onnx.export(self.model, sample_input, path,
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})