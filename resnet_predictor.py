import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.stylegan_model import Generator
from models.models import resnet50
from data.celeba_attrimg_dataset import AttrImgDataset



def train(attr='Male', epochs=20, save_freq=2):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    dataset = AttrImgDataset(attr_pairs=[attr], transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    model = resnet50(2).train().cuda()

    crit = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=1e-3)

    for i in range(1, 1 + epochs):
        acc = 0
        tot = 0
        for j, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = data['img'].cuda()
            label = data['label'].cuda().view(-1)
            logits, prob = model(img)
            loss = torch.nn.functional.cross_entropy(logits, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += label.numel()
            acc += (torch.argmax(prob, dim=1) == label).sum().item()

            if j % 50 == 0:
                with open(f'{attr}_loss.txt', 'a+') as f:
                    f.write(f'epoch: {i}/{epochs}, iter: {j}, loss: {loss:.4f}\n')
        with open(f'{attr}_loss.txt', 'a+') as f:
            f.write(f'epoch: {i}/{epochs}, acc: {acc / tot:.4f}\n')
        if i % save_freq == 0:
            torch.save(model.state_dict(), f'./checkpoints/resnet_predictor/{attr}_{i}.pt')


def test(attr,
         num=10,
         img_size=512,
         save_dir='./datasets/generate_celeba_data',
         stylegan_ckpt='./stylegan_checkpoints/celeba/740000.pt',
         predictor_ckpt='./checkpoints/resnet_predictor/Bangs_10.pt'):
    predictor = resnet50(2).eval().cuda()
    predictor.load_state_dict(torch.load(predictor_ckpt))

    stylegan = Generator(img_size, 512, 8).eval().cuda()
    stylegan.load_state_dict(torch.load(stylegan_ckpt)['g_ema'])
    trunc = stylegan.mean_latent(4096).detach()

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    dataset = AttrImgDataset(attr_pairs=[attr], transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    attr_indices = dataset.get_indices([attr])

    @torch.no_grad()
    def generate_img(latent=None):
        if latent is None:
            latent = stylegan.style(torch.randn(1, 512).cuda())
        img, _, _ = stylegan(
            [latent],
            truncation=0.7,
            truncation_latent=trunc,
            input_is_latent=True,
            randomize_noise=False,
        )
        return img, latent

    
    @torch.no_grad()
    def predict(img):
        logits, probas = predictor(nn.Upsample(128)(img))
        print(probas)
        pred = torch.argmax(probas, dim=1)
        # print(pred)
        return pred

    imgs = []
    preds = []
    for i in range(num):
        img, latent = generate_img()
        pred = predict(img)
        # pred = predict2(img)[:, indices]
        imgs.append(img)
        preds.append(pred.item())
        # pred2 = predict2(img)
        # break
    save_image(torch.cat(imgs,0), 'demo.jpg', nrow=10, normalize=True, range=(-1, 1))
    print(preds)
    '''
    tot = 0
    acc = [0 for _ in attr_pairs]
    for i, data in enumerate(dataloader):
        img = data['img'].cuda()
        label = data['label'].cuda()
        # pred = predict(img).unsqueeze(0)
        pred = predict2(img)[:, indices]
        print(label, pred)

        #tot += 1.
        #for j in range(len(attr_pairs)):
        #    acc[j] += (pred[:, j] == label[:, j]).sum().item()
        #if i > 200:
        #    break
        # print(pred, label, img.shape)
        # save_image(img, 'demo.jpg', normalize=True, range=(-1, 1))
        break
    #print(acc,tot,[x/float(tot) for x in acc])

    '''
ATTR_PAIRS = [
    'Young',
    'Bangs',
    'Male',
    'Smiling',
    'Mouth_Slightly_Open',
    'Big_Lips',
    'Big_Nose',
    'High_Cheekbones',
    'Bags_Under_Eyes',
    'Attractive',
    'Arched_Eyebrows'
]

if __name__ == '__main__':
    attr_pairs = [
        'Young',
        'Bangs',
        'Male',
        'Smiling',
    ]
    # train('Young')
    # train('Bangs')
    # train('Male')
    # train('Smiling')
    # train('Mouth_Slightly_Open')
    # train('Big_Lips')
    # train('Big_Nose')
    # train(attr_pairs)
    test('Bangs')