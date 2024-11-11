import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

# SE 注意力模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Global Average Pooling
        y = x.view(batch, channels, -1).mean(dim=2)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y

# 带有 SEBlock 的卷积块
class ConvBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockWithAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)  # 注意力模块

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)  # 使用 SEBlock
        return x

# Transformer 模块
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio))
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# UNet++ 模块
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels, feature_channels=[64, 128, 256, 512], embed_dim=512, num_heads=8, depth=4):
        super(UNetPlusPlus, self).__init__()
        # 编码器
        self.encoder = nn.ModuleList([
            ConvBlockWithAttention(in_channels if i == 0 else feature_channels[i-1], feature_channels[i])
            for i in range(len(feature_channels))
        ])
        self.pool = nn.MaxPool2d(2)

        # Transformer
        self.flatten = nn.Flatten(2)
        self.transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, depth=depth)

        # 解码器
        self.decoder = nn.ModuleList([
            ConvBlockWithAttention(feature_channels[i], feature_channels[i-1])
            for i in range(len(feature_channels)-1, 0, -1)
        ])
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(feature_channels[i], feature_channels[i-1], kernel_size=2, stride=2)
            for i in range(len(feature_channels)-1, 0, -1)
        ])

        # 输出层
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_features = []
        # 编码器部分
        for enc_block in self.encoder:
            x = enc_block(x)
            encoder_features.append(x)
            x = self.pool(x)

        # Transformer 全局建模
        x = encoder_features[-1]
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # 解码器部分与跳跃连接
        decoder_output = x
        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.decoder)):
            decoder_output = upconv(decoder_output)
            decoder_output = torch.cat((decoder_output, encoder_features[-(i+2)]), dim=1)  # 多级跳跃连接
            decoder_output = dec_block(decoder_output)

        # 输出层
        output = self.final_conv(decoder_output)
        return output

if __name__ == '__main__':
    # 创建一个模型实例
    model = UNetPlusPlus(in_channels=3, out_channels=1)

    # 创建一个示例输入张量 (batch_size=1, in_channels=3, height=256, width=256)
    x = torch.randn(1, 3, 256, 256)

    # 前向传播
    output = model(x)
    print("Output shape:", output.shape)  # 应为 (1, 1, 256, 256)