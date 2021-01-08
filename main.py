from datasets.data import get_dataset
import torchvision

from models.modules import Generator, Discriminator
import torch

from torch.utils.tensorboard import SummaryWriter

config = {
    'batch_size': 32,
    'dataset_root': 'datasets',
    'is_train': True,
    'z_dim': 100,
    'embed_size': 100,
    'lr': 1e-4,
    'channels': 1,
    'num_classes': 10,
    'epochs': 100,
    'img_dim': 28,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("CUDA =", torch.cuda.is_available())

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

train_data = get_dataset(config, transforms=transforms)

gen = Generator(config['z_dim'], config['channels'], img_dim=config['img_dim'], embed_size=config['embed_size'],
                num_classes=config['num_classes']).to(config['device'])
disc = Discriminator(config['channels'], num_classes=config['num_classes'], img_size=config['img_dim']).to(config['device'])

loss_function = torch.nn.BCELoss()

optimiser_G = torch.optim.Adam(gen.parameters(), lr=config['lr'], betas=(0.0, 0.9))
optimiser_D = torch.optim.Adam(disc.parameters(), lr=config['lr'], betas=(0.0, 0.9))

step = 0
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

for epoch in range(config['epochs']):
    for batch_idx, (real, label) in enumerate(train_data):

        real = real.to(config['device'])
        label = label.to(config['device'])

        z = torch.randn((real.shape[0], 100, 1, 1)).to(config['device'])

        score_real = disc(real, label).reshape(-1)

        fake = gen(z, label)
        score_fake = disc(fake, label).reshape(-1)
        lossD = -(torch.mean(score_real) - torch.mean(score_fake))
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimiser_D.step()

        gen_score = disc(fake, label).reshape(-1)
        lossG = -torch.mean(gen_score)

        gen.zero_grad()
        lossG.backward()
        optimiser_G.step()

        if step % 500 == 0:
            print(
                f"Epoch [{epoch}/{config['epochs']}] Batch {batch_idx}/{len(train_data)} \
                                          Loss D: {lossD.item():.4f}, loss G: {lossG.item():.4f}"
            )

            with torch.no_grad():
                fake = gen(z, label)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:config['batch_size']], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:config['batch_size']], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        step += 1