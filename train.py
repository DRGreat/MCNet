import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from models.protonet import ProtoNet
from models.method import Method
from test import test_main, evaluate
from thop import profile, clever_format
from torchinfo import summary
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
import math

logid = time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime(time.time()))


def train(epoch, model, loader, optimizer, lr_scheduler, args=None):
    model.train()
    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):
        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Forward images (way*(shot+query), 3, 84, 84) -> (Size, C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits = model((data_shot, data_query))

        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])
        

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss
        # break
        loss = args.lamb * epi_loss + (1 - args.lamb) * loss_aux

        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(
            f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        # detect_grad_nan(model)
        optimizer.step()
        lr_scheduler.step()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    logfile_path = f"log/{args.dataset}/{args.way}way{args.shot}shot/{logid}"
    if not os.path.exists(logfile_path):
        os.makedirs(logfile_path)
    set_seed(args.seed)
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, math.ceil(len(trainset.data) / args.batch), args.way,
                                      args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    model = Method(args).cuda()
    with open(os.path.join(logfile_path, "log.txt"), "a+") as f:
        f.write(f"{model.__class__.__name__}\n")
        f.write(f"{args}\n\n")
    model = nn.DataParallel(model, device_ids=args.device_ids)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=math.ceil(
        len(trainset.data) / args.batch) * args.max_epoch, eta_min=0)
    max_acc, max_epoch = 0.0, 0
    start = time.time()

    train_losses = []  # 存储训练损失值
    val_losses = []  # 存储验证损失值
    train_accs = []  # 存储训练精度
    val_accs = []  # 存储验证精度

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()
        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, lr_scheduler, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

        train_losses.append(train_loss)  # 记录训练损失值
        val_losses.append(val_loss)  # 记录验证损失值
        train_accs.append(train_acc)  # 记录训练精度
        val_accs.append(val_acc)  # 记录验证精度

        with open(os.path.join(logfile_path, "log.txt"), "a+") as f:
            f.write(f'[train] epo:{epoch:>3} | avg.loss:{train_loss:.4f} | avg.acc:{train_acc:.3f}\n')
            f.write(f'[val] epo:{epoch:>3} | avg.loss:{val_loss:.4f} | avg.acc:{val_acc:.3f}\n\n')

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

    end = time.time()
    with open(os.path.join(logfile_path, "log.txt"), "a+") as f:
        f.write(f"training time: {end - start} seconds\n")

    # 绘制学习曲线图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.max_epoch + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.max_epoch + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.max_epoch + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, args.max_epoch + 1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(logfile_path, 'learning_rate.pdf'),
                format="pdf")
    return model, logfile_path


if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    args.logtest = False
    model, logfile_path = train_main(args)
    args.logtest = True
    # args.dataset = "cub"
    test_acc, test_ci = test_main(model, args, logfile_path)
