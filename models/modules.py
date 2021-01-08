import torch.nn as nn
from collections import OrderedDict
import torch

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_classes, img_size):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.disc = nn.Sequential(OrderedDict([
            ('res_block_1', self.res_block(in_channels+1, 4, 5, 1, 2)),
            ('res_block_2', self.res_block(4, 8, 5, 1, 2)),
            ('flatten', nn.Flatten()),
            ('fc_1', nn.Linear(392, 128)),
            ('fc_relu', nn.LeakyReLU(0.2)),
            ('dropout', nn.Dropout(0.7)),
            ('fc_2', nn.Linear(128, 32)),
            ('fc_relu', nn.LeakyReLU(0.2)),
            ('fc_3', nn.Linear(32, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def forward(self, x, label):
        embedding = self.embed(label).view(label.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)

    def res_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            ('pool', nn.MaxPool2d((2, 2))),
            ('batch_norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.LeakyReLU(0.2))
        ]))


class Generator(nn.Module):
    def __init__(self, z_dim, out_channels, num_classes, img_dim, embed_size, img_size=4):
        super(Generator, self).__init__()
        self.img_dim = img_dim
        self.gen = nn.Sequential(OrderedDict([
            ('upsample_1', self.upsample_block(z_dim+embed_size, img_size*4, 4, 2, 0)),
            ('upsample_2', self.upsample_block(img_size*4, img_size*2, 5, 2, 2)),
            ('upsample_3', self.upsample_block(img_size*2, img_size, 4, 2, 1)),
            ('upsample_4', self.upsample_block(img_size, out_channels, 4, 2, 1)),
            ('tanh', nn.Tanh())
        ]))
        self.embed = nn.Embedding(num_classes, embed_size)

    def upsample_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, label):
        embedding = self.embed(label).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)

        return self.gen(x)


if __name__ == "__main__":
    dis = Discriminator(1, 10, 28)
    import torch

    gen = Generator(100, 1, 10, 28, 20)

    inp = torch.randn((4, 100, 1, 1))
    fake = gen(inp, torch.randint(1, 2, (4,)))
    score = dis(fake, torch.randint(1, 2, (4,)))

    print(score)

