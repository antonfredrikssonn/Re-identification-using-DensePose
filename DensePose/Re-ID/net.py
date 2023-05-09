import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo



param_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


class EmbedNetwork(nn.Module):
    def __init__(self, dims = 128, pretrained_base = True, *args, **kwargs):
        super(EmbedNetwork, self).__init__(*args, **kwargs)
        self.pretrained_base = pretrained_base

        resnet50 = torchvision.models.resnet50()
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = create_layer(64, 64, 3, stride=1)
        self.layer2 = create_layer(256, 128, 4, stride=2)
        self.layer3 = create_layer(512, 256, 6, stride=2)

        self.layer4 = create_layer(1024, 512, 3, stride=1)
        self.fc_G = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed_G = nn.Linear(in_features = 1024, out_features = dims)

        if self.pretrained_base:
            new_state = model_zoo.load_url(param_url)
            state_dict = self.state_dict()
            for k, v in new_state.items():
                if 'fc' in k: continue
                state_dict.update({k: v})
            self.load_state_dict(state_dict)

        for el in self.fc_G.children():
            if isinstance(el, nn.Linear):
                nn.init.kaiming_normal_(el.weight, a=1)
                nn.init.constant_(el.bias, 0)

        nn.init.kaiming_normal_(self.embed_G.weight, a=1)
        nn.init.constant_(self.embed_G.bias, 0)

        self.layerL1 = create_layer(128, 64, 3, stride = 2)
        self.layerL2 = create_layer(128, 64, 3, stride = 2)
        self.layerL3 = create_layer(128, 64, 3, stride = 2)
        self.layerL4 = create_layer(128, 64, 3, stride = 2)
        self.layerL5 = create_layer(128, 64, 3, stride = 2)
        self.layerL6 = create_layer(128, 64, 3, stride = 2)
        self.layerL7 = create_layer(128, 64, 3, stride = 2)
        self.layerL8 = create_layer(128, 64, 3, stride = 2)

        self.fc_L = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed_L = nn.Linear(in_features = 1024, out_features = dims)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        G = self.layer4(x)

        G = F.avg_pool2d(G, G.size()[2:])
        G = G.contiguous().view(-1, 2048)
        G = self.fc_G(G)
        G = self.embed_G(G)

        L = [None]*8
        for i in range(8):
            L[i] = x[:,i*128:(i+1)*128,:,:]

        L[0] = self.layerL1(L[0])
        L[1] = self.layerL2(L[1])
        L[2] = self.layerL3(L[2])
        L[3] = self.layerL4(L[3])
        L[4] = self.layerL5(L[4])
        L[5] = self.layerL6(L[5])
        L[6] = self.layerL7(L[6])
        L[7] = self.layerL8(L[7])

        L = torch.cat((L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7]),1)
        L = F.avg_pool2d(L, L.size()[2:])
        L = L.contiguous().view(-1, 2048)
        L = self.fc_L(L)
        L = self.embed_L(L)
        return G, L


class Bottleneck(nn.Module):
    def __init__(self, in_chan, mid_chan, stride=1, stride_at_1x1=False, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)

        out_chan = 4 * mid_chan
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=stride1x1,
                bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride3x3,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample == None:
            residual = x
        else:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def create_layer(in_chan, mid_chan, b_num, stride):
    out_chan = mid_chan * 4
    blocks = [Bottleneck(in_chan, mid_chan, stride=stride),]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, mid_chan, stride=1))
    return nn.Sequential(*blocks)


class DenseNormReLU(nn.Module):
    def __init__(self, in_feats, out_feats, *args, **kwargs):
        super(DenseNormReLU, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(in_features = in_feats, out_features = out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class ResNet18Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet18Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        
        return out
    
class EmbedNetwork_DSAG(nn.Module):
    def __init__(self, dims = 128):
        super(EmbedNetwork_DSAG, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResNet18Block, 64, 64, 2, stride=2)
        self.layer2 = self._make_layer(ResNet18Block, 64, 128, 2, stride=2)

        self.layerG = self._make_layer(ResNet18Block, 1024, 2048, 2, stride=1)
        self.fc_G = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed_G = nn.Linear(in_features = 1024, out_features = dims)

        self.layerL1 = create_layer(128, 64, 3, stride = 1)
        self.layerL2 = create_layer(128, 64, 3, stride = 1)
        self.layerL3 = create_layer(128, 64, 3, stride = 1)
        self.layerL4 = create_layer(128, 64, 3, stride = 1)
        self.layerL5 = create_layer(128, 64, 3, stride = 1)
        self.layerL6 = create_layer(128, 64, 3, stride = 1)
        self.layerL7 = create_layer(128, 64, 3, stride = 1)
        self.layerL8 = create_layer(128, 64, 3, stride = 1)

        self.fc_L = DenseNormReLU(in_feats = 2048, out_feats = 1024)
        self.embed_L = nn.Linear(in_features = 1024, out_features = dims)

        self.layerL = [None]*8

    def forward(self, input):
        # Run part 1 for 24 DP-images
        x = [None] * 24
        k = 0
        for i in range(4):
            for j in range(6):
                x[k] = input[:,:,i*32:(i+1)*32,j*32:(j+1)*32].to(torch.float).to('cuda')
                k+=1  

        outs = [None]*24
        for i in range(24):
            outs[i] = self.conv1(x[i])
            outs[i] = self.bn1(outs[i])
            outs[i] = self.relu(outs[i])
            outs[i] = self.conv2(outs[i])
            outs[i] = self.relu(outs[i])
            outs[i] = self.layer1(outs[i])

        # Merge round 1
        outs = merge1(outs)

        # Run part 2 for 13 DP-images
        for i in range(13):
            outs[i] = self.layer2(outs[i])

        # Merge round 2
        outs = merge2(outs)

        G = torch.cat((outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], outs[7]),1)
        G = self.layerG(G)
        G = F.avg_pool2d(G, G.size()[2:])
        G = G.contiguous().view(-1, 2048)
        G = self.fc_G(G)
        G = self.embed_G(G)

        L = [None]*8
        L[0] = self.layerL1(outs[0])
        L[1] = self.layerL2(outs[1])
        L[2] = self.layerL3(outs[2])
        L[3] = self.layerL4(outs[3])
        L[4] = self.layerL5(outs[4])
        L[5] = self.layerL6(outs[5])
        L[6] = self.layerL7(outs[6])
        L[7] = self.layerL8(outs[7])

        L = torch.cat((L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7]),1)

        L = F.avg_pool2d(L, L.size()[2:])
        L = L.contiguous().view(-1, 2048)
        L = self.fc_L(L)
        L = self.embed_L(L)
        return G, L

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

def merge1(X):
    merged_X = [None]*13
    merged_X[0], merged_X[1] = X[0], X[1]
    i=2
    for k in range(2,24,2):
        merged_X[i] = X[k] + X[k+1]
        i += 1
    return merged_X

def merge2(X):
    merged_X = [None]*8
    merged_X[0] = X[0] + X[1]
    merged_X[1] = X[2]
    merged_X[2] = X[3]
    merged_X[3] = X[4] + X[5]
    merged_X[4] = X[6] + X[7]
    merged_X[5] = X[8] + X[9]
    merged_X[6] = X[10] + X[11]
    merged_X[7] = X[12]
    return merged_X

class Classifier(nn.Module):
    def __init__(self, in_features=128, num_class=1502):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, num_class)

    def forward(self, x):
        x = self.fc1(x)
        return x



if __name__ == "__main__":
    embed_net = EmbedNetwork(pretrained_base = True)

    in_ten = torch.randn(32, 3, 256, 128)
    out = embed_net(in_ten)
    print(out.shape)