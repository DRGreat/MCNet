import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from models.method import Method


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)
    for i, (data, labels) in enumerate(tqdm_gen, 1):
        print(i)
        print(data.shape)
        print(labels.shape)
        import sys
        sys.exit(0)
    ttt = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            
            data = data.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'cca'

            logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)
            # print(path)
            # print(label)
            # print(torch.argmax(logits, dim=1))
            # print(acc)
            
            # if acc > 70 and acc < 100:
            #     ttt += 1
            #     if ttt == 4:
            #         break

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg()} (curr:{acc:.3f})')
            # if args.logtest:
            #     with open(f"log/{args.dataset}_{args.way}way{args.shot}shot_log{logid}","a+") as f:
            #         f.write(f'[test] | test_loss:{loss} | test_acc:{acc}\n')
            # if acc > 78 and acc < 82:
            #     print("*"*20)
            #     print("acc", acc)
            #     print("*"*20)
            #     args.visualfile = args.visualfile + 1

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args, logid):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args, return_path=False)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=False)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {test_acc} +- {test_ci:.3f}')
    with open(f"log/{args.dataset}_{args.way}way{args.shot}shot_log{logid}","a+") as f:
        f.write(f'[final] epo:{"best":>3} | {test_acc} +- {test_ci:.3f}')

    

    return test_acc, test_ci

logid = time.strftime("%d%H%M%S",time.localtime(time.time()))


if __name__ == '__main__':
    args = setup_run(arg_mode='test')
    args.visualfile = 1
    model_trained_from = "cub"
    args.save_path = os.path.join(f'checkpoints/{model_trained_from}/{args.shot}shot-{args.way}way/', args.extra_dir)
    if model_trained_from == 'miniimagenet':
        args.num_class = 64
    elif model_trained_from == 'cub':
        args.num_class = 100
    elif model_trained_from == 'fc100':
        args.num_class = 60
    elif model_trained_from == 'tieredimagenet':
        args.num_class = 351
    elif model_trained_from == 'cifar_fs':
        args.num_class = 64
    elif model_trained_from == 'cars':
        args.num_class = 130
    elif model_trained_from == 'dogs':
        args.num_class = 70
    
    model = Method(args).cuda()
    # model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    

    test_main(model, args, logid)
