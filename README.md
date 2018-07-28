# torchsummaryX
Improved visualization tool of [torchsummary](https://github.com/sksq96/pytorch-summary). Here, it visualizes kernel size, output shape, # params, and Mult-Adds. Also the torchsummaryX can handle RNN, Recursive NN, or model with multiple inputs.

## Usage
```python
from torchsummaryX import summary
summary(your_model, torch.zeros((1, 3, 224, 224)))
```
Args:
- model (Module): Model to summarize
- x (Tensor): Input tensor of the model with [N, C, H, W] shape dtype and device have to match to the model
- args, kwargs: Other arguments used in `model.forward` function

## Examples
CNN for MNIST
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
summary(Net(), torch.zeros((1, 1, 28, 28)))
```
```
----------------------------------------------------------------------------------------------------
Layer                   Kernel Shape         Output Shape         # Params (K)      # Mult-Adds (M)
====================================================================================================
0_Conv2d               [1, 10, 5, 5]      [1, 10, 24, 24]                 0.26                 0.14
1_Conv2d              [10, 20, 5, 5]        [1, 20, 8, 8]                 5.02                 0.32
2_Dropout2d                        -        [1, 20, 8, 8]                    -                    -
3_Linear                   [320, 50]              [1, 50]                16.05                 0.02
4_Linear                    [50, 10]              [1, 10]                 0.51                 0.00
====================================================================================================
# Params:    21.84K
# Mult-Adds: 0.48M
----------------------------------------------------------------------------------------------------
```

RNN
```python
class Net(nn.Module):
    def __init__(self,
                 vocab_size=20, embed_dim=300,
                 hidden_dim=512, num_layers=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim,
                               num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden
inputs = torch.zeros((100, 1), dtype=torch.long) # [length, batch_size]
summary(Net(), inputs)
```
```
----------------------------------------------------------------------------------------------------
Layer                   Kernel Shape         Output Shape         # Params (K)      # Mult-Adds (M)
====================================================================================================
0_Embedding                [300, 20]        [100, 1, 300]                 6.00                 1.80
1_LSTM                             -        [100, 1, 512]             3,768.32               376.01
  weight_ih_l0           [2048, 300]
  weight_hh_l0           [2048, 512]
  weight_ih_l1           [2048, 512]
  weight_hh_l1           [2048, 512]
2_Linear                   [512, 20]         [100, 1, 20]                10.26                 0.20
====================================================================================================
# Params:    3,784.58K
# Mult-Adds: 378.02M
----------------------------------------------------------------------------------------------------
```

Recursive NN
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1(out)
        return out
summary(Net(), torch.zeros((1, 64, 28, 28)))
```
```
----------------------------------------------------------------------------------------------------
Layer                   Kernel Shape         Output Shape         # Params (K)      # Mult-Adds (M)
====================================================================================================
0_Conv2d              [64, 64, 3, 3]      [1, 64, 28, 28]                36.93                28.90
1_Conv2d              [64, 64, 3, 3]      [1, 64, 28, 28]          (recursive)                28.90
====================================================================================================
# Params:    36.93K
# Mult-Adds: 57.80M
----------------------------------------------------------------------------------------------------
```

Multiple arguments
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x, args1, args2):
        out = self.conv1(x)
        out = self.conv1(out)
        return out
summary(Net(), torch.zeros((1, 64, 28, 28)), "args1", args2="args2")
```
