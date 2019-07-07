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
=================================================================
                Kernel Shape     Output Shape  Params Mult-Adds
Layer
0_conv1        [1, 10, 5, 5]  [1, 10, 24, 24]   260.0    144.0k
1_conv2       [10, 20, 5, 5]    [1, 20, 8, 8]   5.02k    320.0k
2_conv2_drop               -    [1, 20, 8, 8]       -         -
3_fc1              [320, 50]          [1, 50]  16.05k     16.0k
4_fc2               [50, 10]          [1, 10]   510.0     500.0
-----------------------------------------------------------------
                      Totals
Total params          21.84k
Trainable params      21.84k
Non-trainable params     0.0
Mult-Adds             480.5k
=================================================================
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
===========================================================
            Kernel Shape   Output Shape   Params  Mult-Adds
Layer
0_embedding    [300, 20]  [100, 1, 300]     6000       6000
1_encoder              -  [100, 1, 512]  3768320    3760128
2_decoder      [512, 20]   [100, 1, 20]    10260      10240
-----------------------------------------------------------
                       Totals
Total params          3784580
Trainable params      3784580
Non-trainable params        0
Mult-Adds             3776368
===========================================================
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
============================================================
           Kernel Shape     Output Shape   Params  Mult-Adds
Layer
0_conv1  [64, 64, 3, 3]  [1, 64, 28, 28]  36.928k   28901376
1_conv1  [64, 64, 3, 3]  [1, 64, 28, 28]        -   28901376
------------------------------------------------------------
                          Totals
Total params             36.928k
Trainable params         36.928k
Non-trainable params         0.0
Mult-Adds             57.802752M
============================================================
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

Large models with long layer names
```python
import torchvision
model = torchvision.models.resnet18()
summary(model, torch.zeros(4, 3, 224, 224))
```
```
=================================================================================================
                                          Kernel Shape       Output Shape  \
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

                                         Params    Mult-Adds
Layer
0_conv1                                  9.408k  118.013952M
1_bn1                                     128.0         64.0
2_relu                                        -            -
3_maxpool                                     -            -
4_layer1.0.Conv2d_conv1                 36.864k  115.605504M
5_layer1.0.BatchNorm2d_bn1                128.0         64.0
6_layer1.0.ReLU_relu                          -            -
7_layer1.0.Conv2d_conv2                 36.864k  115.605504M
8_layer1.0.BatchNorm2d_bn2                128.0         64.0
9_layer1.0.ReLU_relu                          -            -
10_layer1.1.Conv2d_conv1                36.864k  115.605504M
11_layer1.1.BatchNorm2d_bn1               128.0         64.0
12_layer1.1.ReLU_relu                         -            -
13_layer1.1.Conv2d_conv2                36.864k  115.605504M
14_layer1.1.BatchNorm2d_bn2               128.0         64.0
15_layer1.1.ReLU_relu                         -            -
16_layer2.0.Conv2d_conv1                73.728k   57.802752M
17_layer2.0.BatchNorm2d_bn1               256.0        128.0
18_layer2.0.ReLU_relu                         -            -
19_layer2.0.Conv2d_conv2               147.456k  115.605504M
20_layer2.0.BatchNorm2d_bn2               256.0        128.0
21_layer2.0.downsample.Conv2d_0          8.192k    6.422528M
22_layer2.0.downsample.BatchNorm2d_1      256.0        128.0
23_layer2.0.ReLU_relu                         -            -
24_layer2.1.Conv2d_conv1               147.456k  115.605504M
25_layer2.1.BatchNorm2d_bn1               256.0        128.0
26_layer2.1.ReLU_relu                         -            -
27_layer2.1.Conv2d_conv2               147.456k  115.605504M
28_layer2.1.BatchNorm2d_bn2               256.0        128.0
29_layer2.1.ReLU_relu                         -            -
30_layer3.0.Conv2d_conv1               294.912k   57.802752M
31_layer3.0.BatchNorm2d_bn1               512.0        256.0
32_layer3.0.ReLU_relu                         -            -
33_layer3.0.Conv2d_conv2               589.824k  115.605504M
34_layer3.0.BatchNorm2d_bn2               512.0        256.0
35_layer3.0.downsample.Conv2d_0         32.768k    6.422528M
36_layer3.0.downsample.BatchNorm2d_1      512.0        256.0
37_layer3.0.ReLU_relu                         -            -
38_layer3.1.Conv2d_conv1               589.824k  115.605504M
39_layer3.1.BatchNorm2d_bn1               512.0        256.0
40_layer3.1.ReLU_relu                         -            -
41_layer3.1.Conv2d_conv2               589.824k  115.605504M
42_layer3.1.BatchNorm2d_bn2               512.0        256.0
43_layer3.1.ReLU_relu                         -            -
44_layer4.0.Conv2d_conv1              1.179648M   57.802752M
45_layer4.0.BatchNorm2d_bn1              1.024k        512.0
46_layer4.0.ReLU_relu                         -            -
47_layer4.0.Conv2d_conv2              2.359296M  115.605504M
48_layer4.0.BatchNorm2d_bn2              1.024k        512.0
49_layer4.0.downsample.Conv2d_0        131.072k    6.422528M
50_layer4.0.downsample.BatchNorm2d_1     1.024k        512.0
51_layer4.0.ReLU_relu                         -            -
52_layer4.1.Conv2d_conv1              2.359296M  115.605504M
53_layer4.1.BatchNorm2d_bn1              1.024k        512.0
54_layer4.1.ReLU_relu                         -            -
55_layer4.1.Conv2d_conv2              2.359296M  115.605504M
56_layer4.1.BatchNorm2d_bn2              1.024k        512.0
57_layer4.1.ReLU_relu                         -            -
58_avgpool                                    -            -
59_fc                                    513.0k       512.0k
-------------------------------------------------------------------------------------------------
                            Totals
Total params            11.689512M
Trainable params        11.689512M
Non-trainable params           0.0
Mult-Adds             1.814078144G
=================================================================================================
```
