# quantize Module
import torch
from utils.model_utils import get_model_mb_size

class QuantizeModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, *args, **kwargs):
        x = self.quant(*args, **kwargs)
        x = self.module(x)
        x = self.dequant(x)
        return x

if __name__ == '__main__':
    model_url = ""
    device = torch.device('cuda:0')
    my_module = torch.jit.load(model_url, map_location=device)
    QuantizeModule(my_module)