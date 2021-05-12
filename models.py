import torch
import torch.nn as nn
import torch.nn.functional as F

class SPnet(nn.Module):
    def __init__(self,opts):
        super(SPnet,self).__init__()
        self.in_channel=opts.ncolor
        self.out_channel=opts.ncolor
        if self.in_channel==3:
            self.model=U_net_CBAM_RGB(self.in_channel,self.out_channel)
        else:
            self.model=U_net_CBAM_Spec(self.in_channel,self.out_channel)
    def forward(self,x):
        return self.model(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 1, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 1, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1,kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def convUnit(in_features,out_features):
    unit=nn.Sequential(
        nn.Conv2d(in_channels=in_features,out_channels=out_features,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_features),
        nn.ReLU()
    )

    return unit


class BasicBlock(nn.Module):
    def __init__(self,n_feats):
        super(BasicBlock,self).__init__()
        self.conv=convUnit(n_feats,n_feats)
        self.ca=ChannelAttention(n_feats)
        self.sa=SpatialAttention()
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        res=x
        out=self.conv(x)
        # return out
        out=self.ca(out)*out
        out=self.sa(out)*out
        out=out+res
        out=self.relu(out)
        return out


class U_net_CBAM_Spec(nn.Module):
    def __init__(self,in_colors,out_colors):
        super(U_net_CBAM_Spec,self).__init__()
        feature_nums=[256,512,1024]
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv1=convUnit(in_features=in_colors,out_features=feature_nums[0])
        self.conv_cbam1=BasicBlock(feature_nums[0])

        self.conv2=convUnit(in_features=feature_nums[0],out_features=feature_nums[1])
        self.conv_cbam2=BasicBlock(feature_nums[1])


        self.conv3=convUnit(in_features=feature_nums[1],out_features=feature_nums[2])
        self.conv_cbam3=BasicBlock(feature_nums[2])

        self.conv=nn.Sequential(
            convUnit(1024,2048),
            convUnit(2048,2048)
        )
        self.up1=nn.ConvTranspose2d(in_channels=2048,out_channels=1024,kernel_size=4,stride=2,padding=1)
        self.up2=nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.conv4=convUnit(in_features=2048,out_features=feature_nums[2])
        self.conv_cbam4=BasicBlock(feature_nums[2])

        self.conv5=convUnit(in_features=feature_nums[2],out_features=feature_nums[1])
        self.conv_cbam5=BasicBlock(feature_nums[1])

        self.conv6=convUnit(in_features=feature_nums[1],out_features=feature_nums[0])
        self.conv_cbam6=BasicBlock(feature_nums[0])

        self.final=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=out_colors,kernel_size=3,padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out1=self.conv1(x)
        out1=self.conv_cbam1(out1)
        out1_pool=self.pool(out1)

        out2=self.conv2(out1_pool)
        out2=self.conv_cbam2(out2)
        out2_pool=self.pool(out2)

        out3=self.conv3(out2_pool)
        out3=self.conv_cbam3(out3)
        out3_pool=self.pool(out3)

        outhidden=self.conv(out3_pool)

        out5=self.conv4(torch.cat((self.up1(outhidden),out3),dim=1))
        out5=self.conv_cbam4(out5)

        out6=self.conv5(torch.cat((self.up2(out5),out2),dim=1))
        out6=self.conv_cbam5(out6)

        out7=self.conv6(torch.cat((self.up3(out6),out1),dim=1))
        out7=self.conv_cbam6(out7)

        out=self.final(out7)
        return out


class U_net_Spec(nn.Module):
    def __init__(self,in_colors,out_colors):
        super(U_net_Spec,self).__init__()
        feature_nums = [256, 512, 1024]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = convUnit(in_features=in_colors, out_features=feature_nums[0])
        self.conv_cbam1 = BasicBlock(feature_nums[0])

        self.conv2 = convUnit(in_features=feature_nums[0], out_features=feature_nums[1])
        self.conv_cbam2 = BasicBlock(feature_nums[1])

        self.conv3 = convUnit(in_features=feature_nums[1], out_features=feature_nums[2])
        self.conv_cbam3 = BasicBlock(feature_nums[2])

        self.conv = nn.Sequential(
            convUnit(1024, 2048),
            convUnit(2048, 2048)
        )
        self.up1 = nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.conv4 = convUnit(in_features=2048, out_features=feature_nums[2])
        self.conv_cbam4 = BasicBlock(feature_nums[2])

        self.conv5 = convUnit(in_features=feature_nums[2], out_features=feature_nums[1])
        self.conv_cbam5 = BasicBlock(feature_nums[1])

        self.conv6 = convUnit(in_features=feature_nums[1], out_features=feature_nums[0])
        self.conv_cbam6 = BasicBlock(feature_nums[0])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=out_colors, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv_cbam1(out1)
        out1_pool = self.pool(out1)

        out2 = self.conv2(out1_pool)
        out2 = self.conv_cbam2(out2)
        out2_pool = self.pool(out2)

        out3 = self.conv3(out2_pool)
        out3 = self.conv_cbam3(out3)
        out3_pool = self.pool(out3)

        outhidden = self.conv(out3_pool)

        out5 = self.conv4(torch.cat((self.up1(outhidden), out3), dim=1))
        out5 = self.conv_cbam4(out5)

        out6 = self.conv5(torch.cat((self.up2(out5), out2), dim=1))
        out6 = self.conv_cbam5(out6)

        out7 = self.conv6(torch.cat((self.up3(out6), out1), dim=1))
        out7 = self.conv_cbam6(out7)

        out = self.final(out7)
        return out




