import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.distributed import DistributedSampler

from PIL import Image

# From https://github.com/JiahuiYu/slimmable_networks/blob/4bb2a623f02a183fe08a5b7415338f148f46b363/utils/transforms.py
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

def replace_conv_with_mconv(model):
    for child_name, child in model.named_children():
        classname = child.__class__.__name__
        if classname.find('Conv2d') != -1:
            bias = child.bias is not None
            conv = MaskConv2d(child.in_channels, child.out_channels, child.kernel_size,
                              child.stride, child.padding, child.dilation, child.groups, bias)
            conv.weight = child.weight
            if bias:
                conv.bias = child.bias
            setattr(model, child_name, conv)
        else:
            replace_conv_with_mconv(child)

def replace_mconv_with_conv(model):
    for child_name, child in model.named_children():
        classname = child.__class__.__name__
        if classname.find('MaskConv2d') != -1:
            bias = child.bias is not None
            conv = torch.nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                              child.stride, child.padding, child.dilation, child.groups, bias)
            conv.weight = child.weight
            if bias:
                conv.bias = child.bias
            setattr(model, child_name, conv)
        else:
            replace_mconv_with_conv(child)

class MaskConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

        self.mask = torch.nn.Parameter(torch.ones(out_channels), requires_grad=False)

    def forward(self, input):
        y = torch.nn.functional.conv2d(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        y = (y.permute(0,2,3,1) * self.mask).permute(0,3,1,2)
        return y



class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_dataloader(img_size, dataset, datapath, batch_size, interpolation, no_val, slim_dataaug=False, scale_ratio=0.08, num_gpus=1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    if img_size == 32:
        train_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        val_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(int(time.time()))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        test_set = eval(dataset)(datapath, False, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        if no_val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=8, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=8, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=train_sampler,
                num_workers=8, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=8, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=False
        )
    else:
        if slim_dataaug:
            jitter_param = 0.4
            lighting_param = 0.1
            train_set = eval(dataset)('{}/train'.format(datapath), transform=transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(scale_ratio, 1.0), interpolation=interpolation),
                    transforms.ColorJitter(
                        brightness=jitter_param, contrast=jitter_param,
                        saturation=jitter_param),
                    Lighting(lighting_param),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            train_set = eval(dataset)('{}/train'.format(datapath), transform=transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(scale_ratio, 1.0), interpolation=interpolation),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        test_set = eval(dataset)('{}/val'.format(datapath), transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        sampler = DistributedSampler(train_set) if num_gpus > 1 else None

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=int(batch_size/num_gpus),
            shuffle=(False if sampler else True), sampler=sampler,
            num_workers=8, pin_memory=True, drop_last=True
        )

        sampler = DistributedSampler(test_set) if num_gpus > 1 else None
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(batch_size/num_gpus),
            shuffle=(False if sampler else True), sampler=sampler,
            num_workers=8, pin_memory=True)
        val_loader = test_loader
            
    return train_loader, val_loader, test_loader


def test(model, loader, device='cuda'):
    with torch.no_grad():
        model.eval()

        total = 0.
        top1 = 0.
        top5 = 0.
        bins = {}
        for i, (batch, label) in enumerate(loader):
            batch, label = batch.to(device), label.to(device)
            total += batch.size(0)
            out = model(batch)
            _, top5_pred = out.topk(5, dim=1)
            _, pred = out.max(dim=1)
            top1 += pred.eq(label).sum()
            top5 += top5_pred.eq(label.reshape(-1, 1)).sum()

        total_cost = 0

        return float(top1)/total*100, float(top5)/total*100


def train(model, train_loader, val_loader, teacher=None, hint_layers=None, optimizer=None, epochs=10, steps=None, scheduler=None, run_test=True, name='', device='cuda', no_save=False):
    model.to(device)
    if teacher is not None:
        teacher.eval()

    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)

    # Use number of steps as unit instead of epochs
    if steps:
        epochs = int(steps / len(train_loader)) + 1
        if epochs > 1:
            steps = steps % len(train_loader)

    torch.save({'state_dict': model.state_dict()}, os.path.join('ckpt', '{}_epoch{}.pt'.format(name, 0)))
    for i in range(epochs):
        print('Epoch: {}'.format(i))
        if i == epochs - 1:
            loss = train_epoch(model, train_loader, optimizer, hint_layers=hint_layers, teacher=teacher, steps=steps, device=device)
        else:
            loss = train_epoch(model, train_loader, optimizer, hint_layers=hint_layers, teacher=teacher, device=device)
        if scheduler is not None:
            scheduler.step()

        if run_test:
            acc = test(model, val_loader, eval=True)
            print('Testing Accuracy {:.2f}'.format(acc))
            if i == (epochs-1):
                torch.save({'state_dict': model.state_dict()}, os.path.join('ckpt', '{}_final.pt'.format(name)))
            elif (i+1) % 20 == 0 and not no_save:
                torch.save({'state_dict': model.state_dict()}, os.path.join('ckpt', '{}_epoch{}.pt'.format(name, i+1)))
        sys.stdout.flush()

def train_epoch(model, train_loader, optimizer=None, hint_layers=None, steps=None, device='cuda', teacher=None):
    model.to(device)
    model.train()
    losses = np.zeros(0)
    total_loss = 0
    data_t = 0
    train_t = 0 
    criterion = torch.nn.CrossEntropyLoss()
    if teacher is not None:
        # criterion = torch.nn.MSELoss()
        teacher.to(device)
        teacher.eval()
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        hint_criterion = torch.nn.MSELoss()

    s = time.time()
    for i, (batch, label) in enumerate(train_loader):
        batch, label = batch.to(device), label.to(device)
        data_t += time.time() - s
        s = time.time()

        model.zero_grad()
        output = model(batch)
        if teacher is not None:
            with torch.no_grad():
                t_out = torch.softmax(teacher(batch), dim=1)
            output = F.log_softmax(output, dim=1)
            loss = criterion(output, t_out.detach())
            for (src, tgt) in hint_layers:
                loss += hint_criterion(src.tmp, tgt.tmp)
            loss.backward()
        else:
            loss = criterion(output, label)
            loss.backward()
        optimizer.step()

        total_loss += loss
        losses = np.concatenate([losses, np.array([loss.item()])])

        train_t += time.time() - s
        length = steps if steps and steps < len(train_loader) else len(train_loader)

        if (i % 100 == 0) or (i == length-1):
            print('Training | Batch ({}/{}) | Loss {:.4f} ({:.4f}) | (PerBatchProfile) Data: {:.3f}s, Net: {:.3f}s'.format(i+1, length, total_loss/(i+1), loss, data_t/(i+1), train_t/(i+1)))
        if i == length-1:
            break
        s = time.time()
    return np.mean(losses)

