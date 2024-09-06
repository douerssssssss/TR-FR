import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from s4 import LinearActivation, S4Block as S4


class dialted_residual_inception_module_down(nn.Module):
    def __init__(self, d_model=128, d_input=2, in_channels = 128, out_channels = 128):
        super(dialted_residual_inception_module_down,self).__init__()

        #定义Conv1D层
        self.encoder = nn.Conv1d(d_model,d_model, kernel_size=1)
        #定义encoder层
        #self.encoder = nn.Linear(d_input, d_model)
        #定义S4层
        self.s4 = S4(d_model=d_model)
        #定义激活函数
        self.activation = nn.LeakyReLU(0.3)
        #膨胀卷积
        self.dilated_conv1d_layers1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=4, padding=1, dilation=1)
        self.dilated_conv1d_layers2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=4, padding=2, dilation=2)
        self.dilated_conv1d_layers3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=4, padding=4, dilation=4)
        self.dilated_conv1d_layers4 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=4, padding=8, dilation=8)

        #BarchNormalization
        self.normalize = nn.BatchNorm1d(out_channels)

        self.conv = nn.Conv1d(in_channels=out_channels*4,out_channels=out_channels*2,kernel_size=1)


    def forward(self,x):
        """

        :param x: (B,in_channels, length)
        :return:
        """
        #x = self.encoder(x)
        #x = x.permute(0, 2, 1)
        x,_ = self.s4(x)

        dilated_layer1 = self.dilated_conv1d_layers1(x)
        #dilated_layer1 = self.activation(dilated_layer1)
        #dilated_layer1, _ = self.s4(dilated_layer1)  # (B, H, L) ---> (B, H, L)
        dilated_layer1 = self.normalize(dilated_layer1)

        dilated_layer2 = self.dilated_conv1d_layers2(x)
        dilated_layer2 = self.activation(dilated_layer2)
        #dilated_layer2, _ = self.s4(dilated_layer2)  # (B, H, L) ---> (B, H, L)
        dilated_layer2 = self.normalize(dilated_layer2)


        dilated_layer3 = self.dilated_conv1d_layers3(x)
        dilated_layer3 = self.activation(dilated_layer3)
        #dilated_layer3, _ = self.s4(dilated_layer3)  # (B, H, L) ---> (B, H, L)
        dilated_layer3 = self.normalize(dilated_layer3)


        dilated_layer4 = self.dilated_conv1d_layers4(x)
        dilated_layer4 = self.activation(dilated_layer4)
        #dilated_layer4, _ = self.s4(dilated_layer4)  # (B, H, L) ---> (B, H, L)
        dilated_layer4 = self.normalize(dilated_layer4)


        dilated = torch.cat([dilated_layer1,dilated_layer2,dilated_layer3,dilated_layer4],dim=1)
        #dilated = self.conv(dilated)

        return dilated


class dialted_residual_inception_module_up(nn.Module):
    def __init__(self, d_model=128, d_input=2, in_channels = 128, out_channels = 128):
        super(dialted_residual_inception_module_up,self).__init__()

        #反卷积
        #self.dilated_conv1d_layers4 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=4,output_padding= 1)
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=4, dilation=1, output_padding=1)

        #BarchNormalization
        self.normalize = nn.BatchNorm1d(out_channels)

        self.activation = nn.LeakyReLU(0.3)

        # 定义S4层
        self.s4 = S4(d_model=d_model)


    def forward(self,x):
        """

        :param x: (B,in_channels, length)
        :return:
        """

        #x = self.deconv1(x)
        #x = self.activation(x)
        #x = self.normalize(x)

        dilated_layer1 = self.deconv1(x)
        dilated_layer1 = self.activation(dilated_layer1)
        dilated_layer1 = self.normalize(dilated_layer1)

        x, _ = self.s4(dilated_layer1)  # (B, H, L) ---> (B, H, L)

        return x


class s4_u_net(nn.Module):
    def __init__(self, d_model=64, d_input=2, ):
        super(s4_u_net,self).__init__()

        # 定义encoder层
        self.encoder = nn.Conv1d(d_input,d_model, kernel_size=1)

        self.dilated_block1 = dialted_residual_inception_module_down(d_model=64,in_channels=64,out_channels=64)
        self.dilated_block2 = dialted_residual_inception_module_down(d_model=256,in_channels=256,out_channels=256)
        self.dilated_block3 = dialted_residual_inception_module_up(d_model=256,in_channels=1024,out_channels=256)
        self.dilated_block4 = dialted_residual_inception_module_up(d_model=64,in_channels=256,out_channels=64)

        self.s4 = S4(d_model=1024)

        self.decoder1 = nn.Conv1d(1024,256, kernel_size=1)
        self.decoder2 = nn.Conv1d(256, 1, kernel_size=1)
        self.decoder3 = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self,x):
        x = self.encoder(x.permute(0, 2, 1))  # (B, L, d_input) ---> (B, L, d_model(H))
        x1 = self.dilated_block1(x)
        x2 = self.dilated_block2(x1)

        x3,_ = self.s4(x2)

        #x4 = self.dilated_block3(x3+x2)
        #x5 = self.dilated_block4(x4+x1)

        x_ = self.decoder1(x3)
        x_ = self.decoder2(x_)
        ff = x_.squeeze()

        restrict = lambda ff: ff[..., -1:]
        features = restrict(ff)

        return features






if __name__ == '__main__':

    # 随机生成输入数据
    batch_size = 32

    inputs = torch.range(1,128000,1)
    x = torch.reshape(inputs,shape=(32,2000,2))
    model = s4_u_net()



    # 将输入数据传递给模型进行前向传播
    outputs = model(x)


