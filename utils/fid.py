from torchmetrics.image.fid import FrechetInceptionDistance

def fid(real_images, fake_images):
    frec = FrechetInceptionDistance(normalize=True)
    frec.update(real_images, real=True)
    frec.update(fake_images, real=False)
    return frec.compute()
