import torch
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializes a new Residual Block object
        
        h(x) = f(x) + x
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer,
                for residual block it's recommended to change channel size together with activation map size (size of conv output)
        """
        super(ResBlock, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.downsample = None
        
        if stride > 1:
            self.downsample = torch.nn.Conv2d(in_channels, out_channels, 1, stride=2)
            
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
       
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv2(x)
        
        x = x + residual
        x = self.norm2(x)
        x = torch.nn.functional.leaky_relu(x)
        
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initializes a new Encoder object that encodes input image into latent space (compressed feature representation)
        """
        super(Encoder, self).__init__()
        
        self.activ = torch.nn.functional.leaky_relu
        
        self.incp = torch.nn.Conv2d(3, 64, 5)
        self.norm1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(3, 2)
        self.downsample = torch.nn.MaxPool2d(2, 2)
        
        self.res1 = ResBlock(64, 64)
        self.res2 = ResBlock(64, 64)
        self.res3 = ResBlock(64, 128, 2)
        self.res4 = ResBlock(128, 128)
        self.res5 = ResBlock(128, 128)
        self.res6 = ResBlock(128, 256, 2)
        self.res7 = ResBlock(256, 256)
        self.res8 = ResBlock(256, 256)
        self.res9 = ResBlock(256, 256, 2)
        self.res10 = ResBlock(256, 256)
        self.res11 = ResBlock(256, 256)
        self.res12 = ResBlock(256, 256, 2)
        self.res13 = ResBlock(256, 256)
        self.res14 = ResBlock(256, 512, 2)
        self.res15 = ResBlock(512, 512, 2)
        
        self.avgpool = torch.nn.AvgPool2d(3, 2)
        
        
    def forward(self, x):
        
        x = self.incp(x)
        x = self.norm1(x)
        x = torch.nn.functional.leaky_relu(x)
        
        x = self.maxpool(x)
        
        x = self.activ(self.res1(x) + x)
        x = self.activ(self.res2(x) + x)
        x = self.res3(x)
        x = self.activ(self.res4(x) + x)
        x = self.activ(self.res5(x) + x)
        x = self.res6(x)
        x = self.activ(self.res7(x) + x)
        x = self.activ(self.res8(x) + x)
        x = self.res9(x)
        x = self.activ(self.res10(x) + x)
        x = self.activ(self.res11(x) + x)
        x = self.res12(x)
        x = self.activ(self.res13(x) + x)
        x = self.res14(x)
        x = self.activ(self.res15(x) + self.downsample(x))
       
        x = self.avgpool(x)
        
        x = torch.flatten(x, start_dim=1)
        
        return x
 
class Deconv(torch.nn.Module):
    """
    Initializes a new Deconvolution (transposed convolution) object
    O = (I - 1) x stride - 2 x padding + kernel size
    
    Args:
        in_channels (int): the number of input features of given layer
        out_channels (int): the number of output features of given layer,
            for residual block it's recommended to change channel size together with activation map size (size of conv output)
        kernel = size of the transposed convolution kernel (window)
        stride = size of the transposed convolution stride (step size)
        padding = specifise the usage of the padding (supplement of the image/feature map borders)
        activation = wether use a activation function or not (in last layer it's recommended to not)
    """
    def __init__(self, in_channels, out_channels, stride=1, activation=True):
        super(Deconv, self).__init__()
            
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=1)
        self.upsample = None
        self.activation = activation

        if stride > 1:
            self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            
        self.deconv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, 2, stride=stride, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(out_channels)
        self.norm2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        
        if self.upsample:
            residual = self.upsample(x)
        else:
            residual = x
       
        x = self.deconv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.deconv2(x)
        x = x + residual
        x = self.norm2(x)
        
        if self.activation:
            x = torch.nn.functional.leaky_relu(x)
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        """
        Initializes a new Decoder object that decodes given latent space (compressed feature representation) into image
        """
        self.lin = torch.nn.Linear(512, 512*2*2)
        
        self.dconv1 = Deconv(512, 512, 2)
        self.dconv2 = Deconv(512, 256, 2)
        self.dconv3 = Deconv(256, 256, 1)
        self.dconv4 = Deconv(256, 256, 2)
        self.dconv5 = Deconv(256, 256, 1)
        self.dconv6 = Deconv(256, 256, 1)
        self.dconv7 = Deconv(256, 128, 2)
        self.dconv8 = Deconv(128, 128, 1)
        self.dconv9 = Deconv(128, 128, 1)
        self.dconv10 = Deconv(128, 128, 2)
        self.dconv11 = Deconv(128, 128, 1)
        self.dconv12 = Deconv(128, 64, 2)
        self.dconv13 = Deconv(64, 64, 1)
        self.dconv14 = Deconv(64, 32, 2)
        self.dconv15 = Deconv(32, 3, 2, activation=False)
        
    def forward(self, x):
        
        x = self.lin(x)
        x = x.view(-1, 512, 2, 2)

        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        x = self.dconv5(x)
        x = self.dconv6(x)
        x = self.dconv7(x)
        x = self.dconv8(x)
        x = self.dconv9(x)
        x = self.dconv10(x)
        x = self.dconv11(x)
        x = self.dconv12(x)
        x = self.dconv13(x)
        x = self.dconv14(x)
        x = self.dconv15(x)
 
        return x
    
class Autoencoder(torch.nn.Module):
    """
        Initializes a new Autoencoder object composed of Encoder and Decoder objects to compress and decompress an image
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        
        latent_space = self.encoder(x)
        x = self.decoder(latent_space)
        
        return x, latent_space
    
    
class Inception(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_kernel = [5, 7]):
        """
        Initializes a new Inception block composed of several parallel convolution blocks
        that extract features with different scales and neighbourhood sizes
        
        Args:
            in_channels (int): the number of input features of given layer
            out_channels (int): the number of output features of given layer
            mid_kernel (list of ints): sizes of middle convolution kernels of inception module
        """
        
        super().__init__()
        
        self.branch_left = torch.nn.Conv2d(in_channels, out_channels, 1)
        self.branch_1 = torch.nn.Conv2d(in_channels, out_channels, mid_kernel[0], padding=2)
        self.branch_2 = torch.nn.Conv2d(in_channels, out_channels, mid_kernel[1], padding=3)
        self.branch_right = torch.nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):    
        
        branch_left = self.branch_left(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_right = self.branch_right(x)
        
        out = torch.cat((branch_left, branch_1, branch_2, branch_right), dim=1)
        
        return out
    
    
class MLP(torch.nn.Module):
    
    def __init__(self):
    
        super().__init__()
        
        self.lin1 = torch.nn.Linear(512, 128)
        self.lin2 = torch.nn.Linear(128, 32)
        self.lin3 = torch.nn.Linear(32, 16)
        self.lin4 = torch.nn.Linear(16, 1)
    
    def forward(self, x):    
        
        x = torch.nn.functional.leaky_relu(self.lin1(x))
        x = torch.nn.functional.leaky_relu(self.lin2(x))
        x = torch.nn.functional.leaky_relu(self.lin3(x))
        x = torch.nn.functional.leaky_relu(self.lin4(x))
        
        return x   
