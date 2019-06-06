# torchsummaryX
Improved visualization tool of [torchsummary](https://github.com/sksq96/pytorch-summary). Here, it visualizes kernel size, output shape, # params, and Mult-Adds. Also the torchsummaryX can handle RNN, Recursive NN, or model with multiple inputs.

## Usage
`pip install torchsummaryX` and

```python
from torchsummaryX import summary
summary(your_model, torch.zeros((1, 3, 224, 224)))
```
Args:
- `model` (Module): Model to summarize
- `x` (Tensor): Input tensor of the model with [N, C, H, W] shape dtype and device have to match to the model
- `args, kwargs`: Other arguments used in `model.forward` function

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
========================================================================
                Kernel Shape     Output Shape Params (K) Mult-Adds (M)
Layer
0_conv1        [1, 10, 5, 5]  [1, 10, 24, 24]       0.26         0.144
1_conv2       [10, 20, 5, 5]    [1, 20, 8, 8]       5.02          0.32
2_conv2_drop               -    [1, 20, 8, 8]          -             -
3_fc1              [320, 50]          [1, 50]      16.05         0.016
4_fc2               [50, 10]          [1, 10]       0.51        0.0005
------------------------------------------------------------------------
Params (K):  21.84
Mult-Adds (M):  0.4805
========================================================================
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
==================================================================
            Kernel Shape   Output Shape  Params (K)  Mult-Adds (M)
Layer
0_embedding    [300, 20]  [100, 1, 300]        6.00       0.006000
1_encoder              -  [100, 1, 512]     3768.32       3.760128
2_decoder      [512, 20]   [100, 1, 20]       10.26       0.010240
------------------------------------------------------------------
Params (K):  3784.5800000000004
Mult-Adds (M):  3.7763679999999997
==================================================================
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
===================================================================
           Kernel Shape     Output Shape Params (K)  Mult-Adds (M)
Layer
0_conv1  [64, 64, 3, 3]  [1, 64, 28, 28]     36.928      28.901376
1_conv1  [64, 64, 3, 3]  [1, 64, 28, 28]          -      28.901376
-------------------------------------------------------------------
Params (K):  36.928
Mult-Adds (M):  57.802752
===================================================================
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

```
===================================================================
           Kernel Shape     Output Shape Params (K)  Mult-Adds (M)
Layer                                                             
0_conv1  [64, 64, 3, 3]  [1, 64, 28, 28]     36.928      28.901376
1_conv1  [64, 64, 3, 3]  [1, 64, 28, 28]          -      28.901376
-------------------------------------------------------------------
Params (K):  36.928
Mult-Adds (M):  57.802752
===================================================================
```

Large models with long layer names
```python
import torchvision
model = torchvision.models.resnet18()
summary(model, torch.zeros(4, 3, 224, 224))
```
```
Layer                                                                       
0_conv1                                  [3, 64, 7, 7]  [4, 64, 112, 112]   
1_bn1                                             [64]  [4, 64, 112, 112]   
2_relu                                               -  [4, 64, 112, 112]   
3_maxpool                                            -    [4, 64, 56, 56]   
4_layer1.0.Conv2d_conv1                 [64, 64, 3, 3]    [4, 64, 56, 56]   
5_layer1.0.BatchNorm2d_bn1                        [64]    [4, 64, 56, 56]   
6_layer1.0.ReLU_relu                                 -    [4, 64, 56, 56]   
7_layer1.0.Conv2d_conv2                 [64, 64, 3, 3]    [4, 64, 56, 56]   
8_layer1.0.BatchNorm2d_bn2                        [64]    [4, 64, 56, 56]   
9_layer1.0.ReLU_relu                                 -    [4, 64, 56, 56]   
10_layer1.1.Conv2d_conv1                [64, 64, 3, 3]    [4, 64, 56, 56]   
11_layer1.1.BatchNorm2d_bn1                       [64]    [4, 64, 56, 56]   
12_layer1.1.ReLU_relu                                -    [4, 64, 56, 56]   
13_layer1.1.Conv2d_conv2                [64, 64, 3, 3]    [4, 64, 56, 56]   
14_layer1.1.BatchNorm2d_bn2                       [64]    [4, 64, 56, 56]   
15_layer1.1.ReLU_relu                                -    [4, 64, 56, 56]   
16_layer2.0.Conv2d_conv1               [64, 128, 3, 3]   [4, 128, 28, 28]   
17_layer2.0.BatchNorm2d_bn1                      [128]   [4, 128, 28, 28]   
18_layer2.0.ReLU_relu                                -   [4, 128, 28, 28]   
19_layer2.0.Conv2d_conv2              [128, 128, 3, 3]   [4, 128, 28, 28]   
20_layer2.0.BatchNorm2d_bn2                      [128]   [4, 128, 28, 28]   
21_layer2.0.downsample.Conv2d_0        [64, 128, 1, 1]   [4, 128, 28, 28]   
22_layer2.0.downsample.BatchNorm2d_1             [128]   [4, 128, 28, 28]   
23_layer2.0.ReLU_relu                                -   [4, 128, 28, 28]   
24_layer2.1.Conv2d_conv1              [128, 128, 3, 3]   [4, 128, 28, 28]   
25_layer2.1.BatchNorm2d_bn1                      [128]   [4, 128, 28, 28]   
26_layer2.1.ReLU_relu                                -   [4, 128, 28, 28]   
27_layer2.1.Conv2d_conv2              [128, 128, 3, 3]   [4, 128, 28, 28]   
28_layer2.1.BatchNorm2d_bn2                      [128]   [4, 128, 28, 28]   
29_layer2.1.ReLU_relu                                -   [4, 128, 28, 28]   
30_layer3.0.Conv2d_conv1              [128, 256, 3, 3]   [4, 256, 14, 14]   
31_layer3.0.BatchNorm2d_bn1                      [256]   [4, 256, 14, 14]   
32_layer3.0.ReLU_relu                                -   [4, 256, 14, 14]   
33_layer3.0.Conv2d_conv2              [256, 256, 3, 3]   [4, 256, 14, 14]   
34_layer3.0.BatchNorm2d_bn2                      [256]   [4, 256, 14, 14]   
35_layer3.0.downsample.Conv2d_0       [128, 256, 1, 1]   [4, 256, 14, 14]   
36_layer3.0.downsample.BatchNorm2d_1             [256]   [4, 256, 14, 14]   
37_layer3.0.ReLU_relu                                -   [4, 256, 14, 14]   
38_layer3.1.Conv2d_conv1              [256, 256, 3, 3]   [4, 256, 14, 14]   
39_layer3.1.BatchNorm2d_bn1                      [256]   [4, 256, 14, 14]   
40_layer3.1.ReLU_relu                                -   [4, 256, 14, 14]   
41_layer3.1.Conv2d_conv2              [256, 256, 3, 3]   [4, 256, 14, 14]   
42_layer3.1.BatchNorm2d_bn2                      [256]   [4, 256, 14, 14]   
43_layer3.1.ReLU_relu                                -   [4, 256, 14, 14]   
44_layer4.0.Conv2d_conv1              [256, 512, 3, 3]     [4, 512, 7, 7]   
45_layer4.0.BatchNorm2d_bn1                      [512]     [4, 512, 7, 7]   
46_layer4.0.ReLU_relu                                -     [4, 512, 7, 7]   
47_layer4.0.Conv2d_conv2              [512, 512, 3, 3]     [4, 512, 7, 7]   
48_layer4.0.BatchNorm2d_bn2                      [512]     [4, 512, 7, 7]   
49_layer4.0.downsample.Conv2d_0       [256, 512, 1, 1]     [4, 512, 7, 7]   
50_layer4.0.downsample.BatchNorm2d_1             [512]     [4, 512, 7, 7]   
51_layer4.0.ReLU_relu                                -     [4, 512, 7, 7]   
52_layer4.1.Conv2d_conv1              [512, 512, 3, 3]     [4, 512, 7, 7]   
53_layer4.1.BatchNorm2d_bn1                      [512]     [4, 512, 7, 7]   
54_layer4.1.ReLU_relu                                -     [4, 512, 7, 7]   
55_layer4.1.Conv2d_conv2              [512, 512, 3, 3]     [4, 512, 7, 7]   
56_layer4.1.BatchNorm2d_bn2                      [512]     [4, 512, 7, 7]   
57_layer4.1.ReLU_relu                                -     [4, 512, 7, 7]   
58_avgpool                                           -     [4, 512, 1, 1]   
59_fc                                      [512, 1000]          [4, 1000]   

                                     Params (K) Mult-Adds (M)  
Layer                                                          
0_conv1                                   9.408       118.014  
1_bn1                                     0.128       6.4e-05  
2_relu                                        -             -  
3_maxpool                                     -             -  
4_layer1.0.Conv2d_conv1                  36.864       115.606  
5_layer1.0.BatchNorm2d_bn1                0.128       6.4e-05  
6_layer1.0.ReLU_relu                          -             -  
7_layer1.0.Conv2d_conv2                  36.864       115.606  
8_layer1.0.BatchNorm2d_bn2                0.128       6.4e-05  
9_layer1.0.ReLU_relu                          -             -  
10_layer1.1.Conv2d_conv1                 36.864       115.606  
11_layer1.1.BatchNorm2d_bn1               0.128       6.4e-05  
12_layer1.1.ReLU_relu                         -             -  
13_layer1.1.Conv2d_conv2                 36.864       115.606  
14_layer1.1.BatchNorm2d_bn2               0.128       6.4e-05  
15_layer1.1.ReLU_relu                         -             -  
16_layer2.0.Conv2d_conv1                 73.728       57.8028  
17_layer2.0.BatchNorm2d_bn1               0.256      0.000128  
18_layer2.0.ReLU_relu                         -             -  
19_layer2.0.Conv2d_conv2                147.456       115.606  
20_layer2.0.BatchNorm2d_bn2               0.256      0.000128  
21_layer2.0.downsample.Conv2d_0           8.192       6.42253  
22_layer2.0.downsample.BatchNorm2d_1      0.256      0.000128  
23_layer2.0.ReLU_relu                         -             -  
24_layer2.1.Conv2d_conv1                147.456       115.606  
25_layer2.1.BatchNorm2d_bn1               0.256      0.000128  
26_layer2.1.ReLU_relu                         -             -  
27_layer2.1.Conv2d_conv2                147.456       115.606  
28_layer2.1.BatchNorm2d_bn2               0.256      0.000128  
29_layer2.1.ReLU_relu                         -             -  
30_layer3.0.Conv2d_conv1                294.912       57.8028  
31_layer3.0.BatchNorm2d_bn1               0.512      0.000256  
32_layer3.0.ReLU_relu                         -             -  
33_layer3.0.Conv2d_conv2                589.824       115.606  
34_layer3.0.BatchNorm2d_bn2               0.512      0.000256  
35_layer3.0.downsample.Conv2d_0          32.768       6.42253  
36_layer3.0.downsample.BatchNorm2d_1      0.512      0.000256  
37_layer3.0.ReLU_relu                         -             -  
38_layer3.1.Conv2d_conv1                589.824       115.606  
39_layer3.1.BatchNorm2d_bn1               0.512      0.000256  
40_layer3.1.ReLU_relu                         -             -  
41_layer3.1.Conv2d_conv2                589.824       115.606  
42_layer3.1.BatchNorm2d_bn2               0.512      0.000256  
43_layer3.1.ReLU_relu                         -             -  
44_layer4.0.Conv2d_conv1                1179.65       57.8028  
45_layer4.0.BatchNorm2d_bn1               1.024      0.000512  
46_layer4.0.ReLU_relu                         -             -  
47_layer4.0.Conv2d_conv2                 2359.3       115.606  
48_layer4.0.BatchNorm2d_bn2               1.024      0.000512  
49_layer4.0.downsample.Conv2d_0         131.072       6.42253  
50_layer4.0.downsample.BatchNorm2d_1      1.024      0.000512  
51_layer4.0.ReLU_relu                         -             -  
52_layer4.1.Conv2d_conv1                 2359.3       115.606  
53_layer4.1.BatchNorm2d_bn1               1.024      0.000512  
54_layer4.1.ReLU_relu                         -             -  
55_layer4.1.Conv2d_conv2                 2359.3       115.606  
56_layer4.1.BatchNorm2d_bn2               1.024      0.000512  
57_layer4.1.ReLU_relu                         -             -  
58_avgpool                                    -             -  
59_fc                                       513         0.512  
----------------------------------------------------------------------------------------------------
Params (K):  11689.511999999999
Mult-Adds (M):  1814.0781440000007
====================================================================================================
```
