import torch.onnx
import onnx
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict

def get_state_dict(origin_dict):
    old_keys = origin_dict.keys()
    new_dict = {}

    for ii in old_keys:
        temp_key = str(ii)
        if temp_key[0:7] == "module.":
            new_key = temp_key[7:]
        else:
            new_key = temp_key

        new_dict[new_key] = origin_dict[temp_key]
    return new_dict

input_size = 784
hidden_sizes = [128, 100, 64]
output_size = 10
dropout = 0.0
# model load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_sizes[0])),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
    ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
    ('relu2', nn.ReLU()),
    ('dropout', nn.Dropout(dropout)),
    ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
    ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
    ('relu3', nn.ReLU()),
    ('logits', nn.Linear(hidden_sizes[2], output_size))]))
model.to(device)
model.eval()

checkpoint = torch.load("./checkpoint.pth", map_location=device)
checkpoint_dict = get_state_dict(checkpoint["state_dict"])
model.load_state_dict(checkpoint_dict)

# make dummy data
batch_size = 1
# model input size에 맞게 b c h w 순으로 파라미터 설정
x = torch.randn(batch_size, 784, requires_grad=True).to(device)
# feed-forward test
output = model(x)

# convert
torch.onnx.export(model, x, "./test_onnx.onnx", export_params=True, opset_version=11, do_constant_folding=True
                  , input_names = ['input'], output_names=['output']
                  # , dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
                  )