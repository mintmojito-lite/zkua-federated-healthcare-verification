import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = TinyModel()
dummy = torch.randn(1, 2)

torch.onnx.export(model, dummy, "model.onnx")
print("model.onnx exported successfully!")
