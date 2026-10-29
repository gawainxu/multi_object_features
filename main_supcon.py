from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from datasets import colored_mnist
from datasets import mean_mapping, std_mapping
from datasets import image_size_mapping, classes_groups
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from losses import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='1000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--datasets', type=str, default='colored_mnist', help='dataset')
    parser.add_argument("--protocol_idx", type=int, default=0,)
    parser.add_argument("--class_group_idx", type=int, default=0)
    parser.add_argument('--data_folder', type=str, default="../datasets", help='path to custom dataset')
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo", "MaskCon"], help='choose method')
    parser.add_argument("--method_gama", type=float, default=1.0)
    parser.add_argument("--method_lam", type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')

    opt = parser.parse_args()


    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model
    if opt.method == "SupCon":
        opt.model_name += "_SupCon_protocol_"
    elif opt.method == "SimCLR":
        opt.model_name += "_SimCLR_" + str(opt.method_gama) + "_" + str(opt.method_lam) + "_" + str(opt.temp) + "_protocol_"
    elif opt.method == "SimCLR_CE":
        opt.model_name += "_SimCLR_CE_" + str(opt.method_gama) + "_" + str(opt.method_lam) + "_protocol_"

    opt.model_name += str(opt.protocol_idx)
    opt.model_path = opt.model_path + "_" + str(opt.class_group_idx)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    print(opt.model_name)
    return opt


def set_loader(opt):

    # construct data loader
    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomGrayscale(p=0.2),
                                    normalize])
    transform = TwoCropTransform(transform)
    dataset = colored_mnist(opt=opt, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    return train_loader



def set_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model in ["resnet18", "resnet34", "resnet50"]:
        model = SupConResNet(name=opt.model, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP()
    else:
        model = simCNN_contrastive(opt, in_channels=in_channels)
            
    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available() and opt.use_cuda is True:
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()

    if criterion1 is not None:
        criterion1 = criterion1.cuda()
    if criterion2 is not None:
        criterion2 = criterion2.cuda()
    cudnn.benchmark = True

    linear = None

    return model, linear, criterion1, criterion2


def train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ssl = AverageMeter()
    losses_sup = AverageMeter()

    end = time.time()
    
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]
        
        images = torch.cat([images1, images2], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
       
        bsz = labels.shape[0]
        optimizer.zero_grad()

        if opt.method == 'SupCon':
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss_sup = criterion1(features, labels)
            loss_ssl = loss_sup

        elif opt.method == 'SimCLR':

            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

            loss_ssl = criterion1(features)
            loss_sup = criterion2(features=features, labels=labels)
            loss = opt.method_gama * loss_sup + opt.method_lam * loss_ssl

        elif opt.method == 'SimCLR_CE':

            logits = linear(model.encoder(images))
            labels_linear = torch.cat([labels, labels], dim=0)
            loss_sup = criterion2(logits, labels_linear)
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss_ssl = criterion1(features)
            loss = opt.method_gama * loss_sup + opt.method_lam * loss_ssl

        else:
            raise ValueError('contrastive method not supported: {}'.
                                format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)
        losses_ssl.update(loss_ssl.item())
        losses_sup.update(loss_sup.item())
        # SGD
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  "loss_sup {loss_sup.val:.3f} ({loss_sup.avg:.3f})\t"
                  "loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t".format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_sup=losses_sup, loss_ssl=losses_ssl))
            if opt.method == "SimCLR":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_sup.val:.3f} ({loss_sup.avg:.3f})\t".format(
                        loss_ssl=losses_ssl, loss_sup=losses_sup))
            if opt.method == "SimCLR_CE":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_ce.val:.3f} ({loss_ce.avg:.3f})\t".format(
                        loss_ssl=losses_ssl, loss_ce=losses_sup))
            sys.stdout.flush()

    return (losses.avg, losses_sup.avg, losses_ssl.avg)


def main():

    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())

    # build model and criterion
    model, linear, criterion1, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, epoch, save_file)
    with open(os.path.join(opt.save_folder, "loss"), "wb") as f:
         pickle.dump(losses, f)


if __name__ == '__main__':
    main()
    
