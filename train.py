import torch 
from Dataset import AnimeHumanDataset
import os
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim 
import config 
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator



def train_fn(disc_A, disc_H, gen_H, gen_A, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    A_reals=0
    A_fakes=0
    loop = tqdm(loader, leave=True)

    os.makedirs('D:\\Deanimate\\saved_images', exist_ok=True)

    for idx,(human, anime) in enumerate(loop):
        human=human.to(config.DEVICE)
        anime=anime.to(config.DEVICE)

        #Train Discriminators H and A
        with torch.amp.autocast('cuda',):
            fake_anime=gen_A(human)
            D_A_real = disc_A(anime)
            D_A_fake = disc_A(fake_anime.detach())
            A_reals+=D_A_real.mean().item()
            A_fakes+=D_A_fake.mean().item()
            disc_A_loss_real = mse(D_A_real, torch.ones_like(D_A_real))
            disc_A_loss_fake = mse(D_A_fake, torch.zeros_like(D_A_fake))
            disc_A_loss = disc_A_loss_real + disc_A_loss_fake

            fake_human=gen_H(anime)
            D_H_real = disc_H(human)
            D_H_fake = disc_H(fake_human.detach())
            disc_H_loss_real = mse(D_H_real, torch.ones_like(D_H_real))
            disc_H_loss_fake = mse(D_H_fake, torch.zeros_like(D_H_fake))
            disc_H_loss = disc_H_loss_real + disc_H_loss_fake

            #put it together 
            disc_loss = (disc_A_loss + disc_H_loss)/2


        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #Train Generators H and A
        with torch.amp.autocast('cuda',):
            #Adversarial loss
            D_A_fake = disc_A(fake_anime)
            D_H_fake = disc_H(fake_human)
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))

            #Cycle loss
            cycle_human = gen_H(fake_anime)
            cycle_anime = gen_A(fake_human)
            cycle_human_loss = l1(human, cycle_human)
            cycle_anime_loss = l1(anime, cycle_anime)

            #identity loss
            identity_human = gen_H(human)
            identity_anime = gen_A(anime)
            identity_human_loss = l1(human, identity_human)
            identity_anime_loss = l1(anime, identity_anime)

            # add all together 
            G_loss = (
                loss_G_A
                + loss_G_H
                + cycle_human_loss * config.LAMBDA_CYCLE
                + cycle_anime_loss * config.LAMBDA_CYCLE
                + identity_human_loss * config.LAMBDA_IDENTITY
                + identity_anime_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_human*0.5+0.5,f"saved_images/human_{idx}.png")
            save_image(fake_anime*0.5+0.5,f"saved_images/anime_{idx}.png")

        loop.set_postfix(A_real=A_reals/(idx+1),A_fake=A_fakes/(idx+1))



def main():
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_A,
            gen_A,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A,
            disc_A,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = AnimeHumanDataset(
            root_human=config.TRAIN_DIR + "/Train_human",
            root_anime=config.TRAIN_DIR + "/Train_anime",
            transform=config.transforms,
        )
    
    val_dataset = AnimeHumanDataset(
            root_human=config.VAL_DIR + "/Test_human",
            root_anime=config.VAL_DIR + "/Test_anime",
            transform=config.transforms,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.amp.GradScaler('cuda',)
    d_scaler = torch.amp.GradScaler('cuda',)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_A,
            disc_H,
            gen_H,
            gen_A,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)





if __name__ == "__main__":    
    main()