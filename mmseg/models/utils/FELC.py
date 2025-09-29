import torch
import torch.nn as nn

class MultiDWConv(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        dim1 = dim
        dim = dim // 3

        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv3 = nn.Conv2d(dim, dim, 7, 1, 3, bias=True, groups=dim)

        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim1)

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.dwconv1(x1)
        x2 = self.dwconv2(x2)
        x3 = self.dwconv3(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.act(self.bn(x))
        return x
        
class FELC(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x