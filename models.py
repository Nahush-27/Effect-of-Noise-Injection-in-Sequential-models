import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config
from typing import Tuple
from torchvision import datasets, transforms
import copy
import numpy as np
import matplotlib.pyplot as plt

class MNISTNet1(nn.Module):
    def __init__(self):
        super(MNISTNet1, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)

class MNISTNet2(nn.Module):
    def __init__(self):
        super(MNISTNet2, self).__init__()
        self.fc1 = nn.Linear(28*28, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)

class MNISTNet3(nn.Module):
    def __init__(self):
        super(MNISTNet3, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(self.fc6(x), dim=1)

class CIFARCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFARCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding = 1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x))) 
        x = x.mean(dim=[2,3]) 
        x = self.fc1(x)
        return x

class CIFARCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding = 1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.fc1 = nn.Linear(128, 5000)
        self.fc2 = nn.Linear(5000, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x))) 
        x = x.mean(dim=[2,3]) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x       

### CIFAR 100 stuff ###
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))

##### REs CIFAR 100

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if False:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out_final = self.linear(out)
        return out_final 
    
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNet200():
    return ResNet(Bottleneck, [3, 24, 36, 3])

def ResNet270():
    return ResNet(Bottleneck, [3,36,48,3])

def ResNet336():
    return ResNet(Bottleneck, [3,44,62,3])

def ResNet500():
    return ResNet(Bottleneck, [3,70,90,3])

#Problem 3
class LSTM(nn.Module):

  def _init_( self, inSize , hiddenSize, nLayers ):
    super()._init_()

    self.inSize = inSize
    self.hiddenSize = hiddenSize
    self.nLayers = nLayers

    self.LSTM = nn.LSTM ( input_size = self.inSize,
                          hidden_size = self.hiddenSize,
                          num_layers = self.nLayers,
                          batch_first = True
                        )

    self.fc1 = nn.Linear( in_features = self.hiddenSize, out_features = 40)
    # Fully connected Layer, takes output of LSTM, has 40 neuron outputs
    self.fc2 = nn.Linear(in_features = 40, out_features =1)
    # 2nd FC, takes o/p of fc1, has one nueron output to be used for prediction

    self.relu = nn.ReLU() #Rectified Linear Unit act. fun

  def forward( self, x):

    # Define the hidden and cell states of LSTM (have size = (layers X batchsize X hiddensize)
    # since its not stackedRNN hence layers = 1
    # batchsize in the number of windows i.e, Xtrain.size(0)
    # and these states are initialized to 0 )

    h0 = torch.zeros ( self.nLayers, x.size(0), self.hiddenSize,device=x.device ).float()
    c0 = torch.zeros ( self.nLayers, x.size(0), self.hiddenSize,device=x.device ).float()

    _ , ( hout , _ ) = self.LSTM( x, ( h0, c0 ) )
    # The input format of LSTM is (inputdata, (h_0, c_0))
    # The output format of LSTM is (outputdata, (h_n, c_n))
    # We are not interested in outputdata, which packs the output of LSTM(h_t) at each timestep
    # Note that here we only require final hidden state of LSTM at the last stage i.e, h_n hence we ignore others as _

    hout = hout.view( -1, self.hiddenSize )

    # View works similar to reshape but The new tensor will always share its data with the original tensor.
    # This means that if you change the original tensor, the reshaped tensor will change and vice versa.
    # if we give -1, PyTorch should infer the size of that dimension based on the size of the other dimensions
    # and the total number of elements in the tensor.
    # We did this since we are going to pass hout to fc1 which expects 'hiddenSize' input

    out = self.fc2( self.relu( self.fc1( hout ) ) )

    return out
  



# Problem 1
class ImdbNet1(nn.Module):
    def __init__(self, word_vec, emb_dim, num_heads, hidden_dim, num_layers, num_classes, device, dropout=0.1):
        super(ImdbNet1, self).__init__()
        self.word_vec = word_vec.to(device)
        self.emb_dim = emb_dim
        self.positional_encoding = PositionalEncoding(emb_dim, dropout, device=device)
        encoder_layer = nn.TransformerEncoderLayer(emb_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        #sent_size = x.size(1)
        embedded = self.word_vec[x]
        embedded = self.positional_encoding(embedded) #.view(-1, sent_size, self.emb_dim)
        output = self.transformer_encoder(embedded)
        output = output.mean(dim=1)  # Global average pooling
        output = self.fc(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=torch.device("cuda")):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term1= torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        div_term2= torch.exp(torch.arange(0, d_model-1, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term1.to(device))
        pe[:, 1::2] = torch.cos(position * div_term2.to(device))
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaleNorm(nn.Module):
    """Represents Scale Norm layer

    reference: https://github.com/tnq177/transformers_without_tears
    """
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        scale = nn.Parameter(torch.tensor(scale))
        self.register_buffer("eps", torch.Tensor([eps]))
        self.register_parameter("scale", scale)

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):
    """Represents the encoder of the "Attention is All You Need" Transformer.

    Original architecture does not contain 'ScaleNorm' layers.
    I have used them since they improve the convergence of the model.
    """

    def __init__(self, num_layers, num_heads, d_model, ff_dim, p_dropout):
        """Initializes the module.

        Arguments:
            num_layers (int): Number of Transformer encoder layers
            num_heads (int): Number of self-attention heads per layer
            d_model (int): Embedding dimension of every token
            ff_dim (int): Number of neurons of middle layer in the feedgforward segment
            p_dropout (float): Probability used for dropout layers
        """
        super(TransformerEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(num_heads, d_model, ff_dim, p_dropout) for _ in range(num_layers)
        ])
        self.scale_norm = ScaleNorm(d_model ** 0.5)

    def forward(self, x, padd_mask=None):
        """Performs forward pass of the module."""
        attn_weights_accumulator = []
        for encoder_layer in self.encoder_blocks:
            x, attn_weights = encoder_layer(x, padd_mask)
            attn_weights_accumulator.append(attn_weights)

        x = self.scale_norm(x)
        return x, attn_weights_accumulator


class TransformerEncoderLayer(nn.Module):
    """Represents a single Transformer Encoder Block.

    Original architecture does not contain 'ScaleNorm' layers.
    I have used them since they improve the convergence of the model.
    """

    def __init__(self, num_heads, d_model, ff_dim, p_dropout):
        """Initializes the module.

        Arguments:
            num_heads (int): Number of self-attention heads
            d_model (int): Embedding dimension of every token
            ff_dim (int): Number of neurons of middle layer in the feedgforward segment
            p_dropout (float): Probability used for dropout layers
        """
        super(TransformerEncoderLayer, self).__init__()
        self.scale_norm_1 = ScaleNorm(d_model ** 0.5)
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, p_dropout)
        self.dropout_1 = nn.Dropout(p_dropout)

        self.scale_norm_2 = ScaleNorm(d_model ** 0.5)
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout_3 = nn.Dropout(p_dropout)

    def forward(self, x, padd_mask=None):
        """Performs forward pass of the module."""
        x = self.scale_norm_1(x)
        skip_connection = x
        attn_output, attn_weights = self.multi_head_attention(query=x, key=x, value=x, padd_mask=padd_mask)
        x = skip_connection + self.dropout_1(attn_output)

        x = self.scale_norm_2(x)
        skip_connection = x
        x = self.ff_net(x)
        x = skip_connection + self.dropout_3(x)

        return x, attn_weights


class MultiHeadAttention(nn.Module):
    """Represents Multi-Head Attention Module"""

    def __init__(self, num_heads, d_model, p_dropout):
        """Initializes the module.

        Arguments:
            num_heads (int): Number of attention heads
            d_model (int): Embedding dimension of each input token
            p_dropout (float): Probability used for dropout layers
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % num_heads) == 0, "Embedding dimension d_model must be divisible by the number of heads!"
        self.d_head = int(d_model // num_heads)
        self.num_heads = num_heads
        # Represents query, key, value matrices used for mapping the input sequence
        self.qkv_matrices = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

        self.dropout = nn.Dropout(p_dropout)
        self.out_projection = nn.Linear(d_model, d_model)

    def self_attention(self, query, key, value, padd_mask=None):
        """Performs scaled dot-product attention from 'Attention is All You Need'.

        Arguments:
            query (torch.Tensor): Query vector. Expected shape: (seq_len, batch_size, embedding_dim)
            key (torch.Tensor): Key vector. Expected shape: (seq_len, batch_size, embedding_dim)
            value (torch.Tensor): Value vector. Expected shape: (seq_len, batch_size, embedding_dim)
            padd_mask (torch.Tensor): Expected shape: (batch_size, seq_len)
                Usage: Specifies if some tokens should be ignored when calculating attention scores
        Returns:
            output (torch.Tensor): Represents attention combination of input tensors.
                Expected shape: (seq_len, batch_size, embedding_dim)
            attn_weights (torch.Tensor): Attention weights for each token
        """
        seq_len, batch_size, d_model = query.shape
        if padd_mask is not None:
            assert padd_mask.shape == (batch_size, seq_len), f"Invalid mask shape! Expected shape of ({batch_size}, {seq_len})"
            padd_mask = padd_mask.view(batch_size, 1, 1, seq_len). \
                expand(-1, self.num_heads, -1, -1). \
                reshape(batch_size * self.num_heads, 1, seq_len)

        # We map the order of dimensions to (bsz * head_dim, seq_len, d_head)
        query = query.contiguous().view(seq_len, batch_size * self.num_heads, self.d_head).transpose(0, 1)
        key = key.contiguous().view(seq_len, batch_size * self.num_heads, self.d_head).transpose(0, 1)
        value = value.contiguous().view(seq_len, batch_size * self.num_heads, self.d_head).transpose(0, 1)

        # Scores shape: (bsz * head_dim, seq_len, seq_len)
        attn_scores = torch.bmm(query, key.transpose(-2, -1))
        if padd_mask is not None:
            attn_scores.masked_fill_(padd_mask == torch.tensor(True), float("-inf"))
        attn_scores /= (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # Output shape: (bsz * head_dim, seq_lean, d_head)
        output = torch.bmm(attn_weights, value)
        # Map the output to the original input shape
        output = output.transpose(1, 0).contiguous().view(seq_len, batch_size, d_model)
        return output, attn_weights

    def forward(self, query, key, value, padd_mask=None):
        """Performs forward pass of the module"""
        # Map the input into query key and value
        query, key, value = [mapper_net(input_vec) for mapper_net, input_vec in zip(self.qkv_matrices, [query, key, value])]
        # Perform multi-head self-attention
        attn_output, attn_weights = self.self_attention(query, key, value, padd_mask)
        output = self.out_projection(attn_output)

        return output, attn_weights


# Problem 2

class BiLSTMNER(nn.Module):
    def __init__(self, word_vec, emb_dim, hidden_dim, num_layers, num_classes, device, dropout=0.1):
        super(BiLSTMNER, self).__init__()
        self.word_vec = word_vec.to(device)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(self.word_vec)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bidirectional LSTM, so *2

    def forward(self, x):
        embedded = self.word_vec[x]
        embedded = self.dropout(embedded)
        
        lstm_output, _ = self.lstm(embedded)
        
        # Concatenate the hidden states of both directions
        lstm_output = torch.cat((lstm_output[:, :, :self.hidden_dim],
                                 lstm_output[:, :, self.hidden_dim:]), dim=2)
        
        # Apply fully connected layer
        output = self.fc(lstm_output)
        return output.view(-1, self.num_classes, output.size(1))