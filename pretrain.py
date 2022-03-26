import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run,by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from models.method import Method
from test import test_main, evaluate

logid = time.strftime("%d%H%M%S",time.localtime(time.time()))

def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            labels = labels.cuda()

            model.module.mode = 'encoder'
            data = model(data)
            model.module.mode = 'fc'
            logits = model(data)
            loss = F.cross_entropy(logits, labels)
            acc = compute_accuracy(logits, labels)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()




def train(epoch, model, train_loader_aux, optimizer, args=None):
    model.train()

    loss_meter = Meter()
    acc_meter = Meter()

    tqdm_gen = tqdm.tqdm(train_loader_aux)

    for i, (data_aux, train_labels_aux) in enumerate(tqdm_gen, 1):

        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN


        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss = F.cross_entropy(logits_aux, train_labels_aux)

        acc = compute_accuracy(logits_aux, train_labels_aux)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    Dataset = dataset_builder(args)


    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=False)


    valset = Dataset('val', args)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch, num_workers=8, pin_memory=False)
    ''' fix val set for all epochs '''
    val_loader = [x for x in val_loader]

    set_seed(args.seed)
    # model = RENet(args).cuda()
    model = Method(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if not args.no_wandb:
        wandb.watch(model)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loader_aux, optimizer, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')
        with open(f"miniimagenet_{args.way}way{args.shot}shot_log{logid}","a+") as f:
            f.write(f'[train] epo:{epoch:>3} | avg.loss:{train_loss:.4f} | avg.acc:{train_acc:.3f}\n')
            f.write(f'[val] epo:{epoch:>3} | avg.loss:{val_loss:.4f} | avg.acc:{val_acc:.3f}\n\n')


        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'pretrain.pth'))


        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
