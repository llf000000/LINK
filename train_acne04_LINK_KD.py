import random
import numpy as np
import torch
import os
def seed_torch(seed=42):
    random.seed(seed) # Python的随机性
    os.environ["PYTHONHASHSEED"] = str(seed) # 设置python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed) # numpy的随机性
    torch.manual_seed(seed) # torch的CPu随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed) # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPu.torch的GPu随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be Falsetorch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.deterministic = True # 选择确定性算法
seed=42
seed_torch(seed)

import sys
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset import dataset_processing
from transforms.affine_transforms import *
from transforms.affine_transforms import RandomRotate
from torch.utils.data import DataLoader
from utils.report import report_precision_se_sp_yi
from utils.utils import *
import time
import shutil
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from models.LINK.resnet34.custom_resnet34_LINK import CustomResNet34
from models.LINK.densenet121.custom_densenet121_LINK import CustomDenseNet121
os.environ['TORCH_HOME'] = os.getcwd()

def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
    
def parse_args():
    import argparse

    # 获取当前时间
    global current_time
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    # Hyper Parameters
    parser = argparse.ArgumentParser(description='Acne Classification') 
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('--batch-size-test', type=int, default=32, help='batch_size_test') 
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--num-classes', type=int, default=4, help='num_classes')  
    parser.add_argument('--current-time',  type=str, default=current_time, help='current_time')                                                                
    parser.add_argument('--data-path', type=str, default='/root/workspace/dataset/acne04/JPEGImages', help='dataset path')
    parser.add_argument('--cross-val-lists', nargs='+', default=['2'], help='每个字符串代表一个交叉验证的索引或标识符。') # ['0', '1', '2', '3', '4']
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34', help='model architecture: ')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dataset', default='acne04', help='dataset setting')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument("--M", default="[1,2,4]",type=str,help='M')
    parser.add_argument('--distance_weight', default=100, type=int)
    parser.add_argument('--angle_weight', default=80, type=int)
    parser.add_argument('--eps', default=1e-12, type=float)
    parser.add_argument('--squared', default=False, type=bool, help='Flag to indicate squaring operation')
    parser.add_argument('--ce_loss_weight', default=1., type=float)
    parser.add_argument('--ii_skd_loss_weight', default=1., type=float)
    parser.add_argument('--kd_loss_weight', default=1., type=float)
    parser.add_argument('--temperature', default=4, type=int)
    parser.add_argument('--teacher_weight_root', default="/root/workspace/baseline_Medical/baseline_ACNE04/checkpoint/pair/densenet121/teacher/", type=str)
    parser.add_argument("--download_size", default=224,type=int,help='img_size')

    args = parser.parse_args()
    return args

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.split('checkpoint.pth.tar')[0]+'model_best.pth.tar')

def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch % 30 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
        return optimizer.param_groups[0]['lr']
    return optimizer.param_groups[0]['lr']

def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    # f_s, f_t: [B, N, C]
    B, N, C = f_s.shape

    # 距离损失
    with torch.no_grad():
        t_d = torch.cdist(f_t.contiguous(), f_t.contiguous(), p=2 if not squared else 1)
        mask = t_d > 0
        sum_td = (t_d * mask).sum(dim=(1,2))
        count_td = mask.sum(dim=(1,2))
        mean_td = sum_td / (count_td + eps)
        mean_td = mean_td.view(-1, 1, 1)
        t_d = t_d / (mean_td + eps)

    d = torch.cdist(f_s.contiguous(), f_s.contiguous(), p=2 if not squared else 1)
    mask = d > 0
    sum_d = (d * mask).sum(dim=(1,2))
    count_d = mask.sum(dim=(1,2))
    mean_d = sum_d / (count_d + eps)
    mean_d = mean_d.view(-1, 1, 1)
    d = d / (mean_d + eps)

    loss_d = F.smooth_l1_loss(d, t_d, reduction='none')  # [B, N, N]
    loss_d = loss_d.mean(dim=(1,2))  # [B]

    # 角度损失
    with torch.no_grad():
        td = f_t.unsqueeze(2) - f_t.unsqueeze(1)  # [B, N, N, C]
        norm_td = F.normalize(td, p=2, dim=3)
        t_angle = torch.matmul(norm_td, norm_td.transpose(2, 3)).view(B, -1)  # [B, N*N]

    sd = f_s.unsqueeze(2) - f_s.unsqueeze(1)  # [B, N, N, C]
    norm_sd = F.normalize(sd, p=2, dim=3)
    s_angle = torch.matmul(norm_sd, norm_sd.transpose(2, 3)).view(B, -1)  # [B, N*N]

    loss_a = F.smooth_l1_loss(s_angle, t_angle, reduction='none')  # [B, N*N]
    loss_a = loss_a.mean(dim=1)  # [B]

    loss = distance_weight * loss_d + angle_weight * loss_a  # [B]
    return loss  # 返回每个样本的loss

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd

def multi_rkd_per_sample(out_s_multi, out_t_multi, distance_weight=25, angle_weight=50):
    # out_s_multi, out_t_multi: [B, N, C]
    loss = rkd_loss(
        out_s_multi, out_t_multi,
        squared=False, eps=1e-12,
        distance_weight=distance_weight,
        angle_weight=angle_weight
    )
    return loss  # [B]

accuracy_total = 0

def main(cross_val_index,args):
    seed_torch(seed)
    global accuracy_total

    '''从以下内容开始读取ACNE04'''
    TRAIN_FILE = '/root/workspace/dataset/acne04/NNEW_trainval_' + cross_val_index + '.txt'
    TEST_FILE = '/root/workspace/dataset/acne04/NNEW_test_' + cross_val_index + '.txt'
    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])
    
    train_dataset = dataset_processing.DatasetProcessing(
        args.data_path, TRAIN_FILE, transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomRotate(rotation_range=20),
                normalize,
            ])) # return img, label, lesion

    val_dataset = dataset_processing.DatasetProcessing(
        args.data_path, TEST_FILE, transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
    
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # 计算各类别数量
    cls_num_list = [0] * args.num_classes  # NUM_CLASSES 是你数据集中的类别数量

    for label in train_dataset.labels:
        cls_num_list[label] += 1

    train_cls_num_list = np.array(cls_num_list)
    args.cls_num_list = cls_num_list
    
    num_classes = args.num_classes

    log.write("Class Counts:", cls_num_list)
    print("Class Counts:", cls_num_list)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              sampler=None,
                              pin_memory=False) # 将训练数据集封装为一个可迭代的批处理对象

    val_loader = DataLoader(val_dataset,
                             batch_size=args.batch_size_test,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=False)


    '''以上内容开始读取ACNE04'''
    teacher = CustomDenseNet121(pretrained=False,num_classes=4,M = args.M)
    pretrain_model_path = args.teacher_weight_root + 'cross_val_index_' + cross_val_index + '_model_best.pth'
    teacher.load_state_dict(load_checkpoint(pretrain_model_path)["state_dict"])
    teacher.to(args.device)

    # student = torchvision.models.resnet34(pretrained=True)
    student = CustomResNet34(pretrained=True,num_classes=args.num_classes,M=args.M)
    _ = print_model_param_nums(model=student)

    student.to(args.device)
    # construct an optimizer
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    # save_path = './weights/resNet32_best.pth'
    train_steps = len(train_loader)
    best_report = ""
    best_epoch = 0
    for epoch in range(args.epochs):
        best_acc1 = 0
        losses_train = AverageMeter('Loss', ':.4e')
        # train
        running_loss = 0.0
        adjust_learning_rate(epoch, optimizer)

        train_bar = tqdm(train_loader, file=sys.stdout)
        student.train()

        for step, data in enumerate(train_bar):
            images, labels = data

            logits_student, _, patch_s  = student(images.to(args.device))

            with torch.no_grad():
                logits_teacher, _, patch_t = teacher(images.to(args.device))

            loss_ce = F.cross_entropy(logits_student, labels.to(args.device))
            loss_kd = kd_loss(logits_student, logits_teacher, temperature=args.temperature)
            # 遍历每个样本，计算其21个区域的关系损失
            # 调整 patch_s 和 patch_t 的形状以适应 multi_rkd_per_sample 函数
            patch_s = patch_s.permute(0, 2, 1)  # 变成 (batch_size, 21, channels)
            patch_t = patch_t.permute(0, 2, 1)  # 变成 (batch_size, 21, channels)
            # 扩展教师的全局样本输出，以匹配学生的 21 个区域

            # 遍历每个样本，计算其21个区域的RKD损失
            batch_loss_kd = 0
            losses = multi_rkd_per_sample(
                patch_s, patch_t,
                distance_weight=args.distance_weight,
                angle_weight=args.angle_weight
            )  # [B]
            rkd_loss = losses.mean()
            # 计算整个批次的平均RKD损失
            loss = args.ii_skd_loss_weight*rkd_loss + args.kd_loss_weight*loss_kd + args.ce_loss_weight* loss_ce
            
            losses_train.update(loss.item(), images[0].size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss.item())

        # validate
        student.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top4 = AverageMeter('Acc@4', ':6.2f')
            eps = np.finfo(np.float64).eps

            y_true = np.array([])
            y_pred = np.array([])

            all_preds = []
            all_targets = []

            end = time.time()

            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images.to(args.device)
                y_true = np.hstack((y_true, val_labels.data.cpu().numpy()))
                
                outputs,_,_ = student(val_images.to(args.device))

                # measure accuracy and record loss
                acc1, acc4 = accuracy(outputs, val_labels.to(args.device), topk=(1, 4))
                losses.update(loss.item(), val_images.size(0))
                top1.update(acc1.item(), val_images.size(0))
                top4.update(acc4.item(), val_images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(val_labels.cpu().numpy())

                predict_y = torch.max(outputs, dim=1)[1]
                y_pred = np.hstack((y_pred, predict_y.data.cpu().numpy()))

                acc += torch.eq(predict_y, val_labels.to(args.device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)

        val_accurate = acc / val_num
        log.write('[epoch %d] train_loss: %.3f  val_accuracy: %.3f\n' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        
        # 计算每个类的准确率
        flag = 'val'
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(cls_hit, cls_cnt, out=np.zeros_like(cls_hit, dtype=float), where=cls_cnt!=0)
        output_prec = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@4 {top4.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top4=top4, loss=losses))
        
        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list > 20)
        few_shot = train_cls_num_list <= 20

        # Calculate average accuracy for each category
        # 添加长度检查和条件判断
        if len(many_shot) == len(cls_acc):
            many_avg = np.sum(cls_acc[many_shot]) * 100 / (np.sum(many_shot) + eps)
        else:
            print("Length mismatch for many_shot:", len(many_shot), len(cls_acc))
            many_avg = 0  # 或者使用其他默认值

        if len(medium_shot) == len(cls_acc):
            med_avg = np.sum(cls_acc[medium_shot]) * 100 / (np.sum(medium_shot) + eps)
        else:
            print("Length mismatch for medium_shot:", len(medium_shot), len(cls_acc))
            med_avg = 0  # 或者使用其他默认值

        if len(few_shot) == len(cls_acc):
            few_avg = np.sum(cls_acc[few_shot]) * 100 / (np.sum(few_shot) + eps)
        else:
            print("Length mismatch for few_shot:", len(few_shot), len(cls_acc))
            few_avg = 0  # 或者使用其他默认值

        output_count = "many avg, med avg, few avg, Overall accuracy (All): {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(many_avg, med_avg, few_avg, top1.avg)
        
        # 计算Top-1和Top-5错误率
        top1_error = 1 - top1.avg / 100
        top5_error = 1 - top4.avg / 100

        output_error = f'     Top-1 Error: {top1_error:.3f} ' + f'Top-5 Error: {top5_error:.3f}'
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        
        _, AVE_ACC, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true) # Result, AVE_ACC, report

        is_best = False
        if val_accurate > best_acc:
            is_best = True
            best_acc = val_accurate
            best_report = pre_se_sp_yi_report + "\n"
            best_report = best_report + output_prec + "   "+ output_error + "\n"
            best_report = best_report + out_cls_acc + "\n"
            best_report = best_report + output_count + "\n" + "\n"
            best_epoch = epoch + 1

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': student.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            '{}/cross_val_index_{}_checkpoint.pth.tar'.format(args.folder_path,cross_val_index))


        if True:
            log.write(str(pre_se_sp_yi_report) + '\n')
            log.write(output_prec + '   ' + output_error + '\n')
            log.write(out_cls_acc + '\n')
            log.write(output_count+ '\n'+'\n')
            log.write("best result has happened on the {} epoch until now: \n".format(best_epoch))
            log.write(best_report + '\n')
            log.flush()

    accuracy_total += best_acc
    log.write("best result happened on the {} epoch: \n".format(best_epoch))
    log.write(best_report + '\n')

    log.write('Finished Training')

# 获取当前脚本的文件名
current_script_name = os.path.basename(__file__)

if __name__ == '__main__':
    seed_torch(seed)

    args = parse_args()
    log = Logger()
    
    # 创建文件夹
    args.folder_path = './checkpoint/{}'.format(current_time)+'_'+'_'.join([args.dataset, current_script_name.split(".")[0], "/"])
    if not os.path.exists(args.folder_path):
        os.mkdir(args.folder_path)

    args.log_file_name = args.folder_path+args.current_time+'.log'
    log.open(args.log_file_name, mode="a")

    # 在日志中写入当前文件名
    log.write(f"Current script: {current_script_name}\n") 
    
    
    for _argsk, _argsv in args._get_kwargs():
        message = '--{} {}\n'.format(_argsk, _argsv)
        log.write(message) # 遍历所有命令行参数，打印出每个参数的名称和值。

    for cross_val_index in args.cross_val_lists:
        log.write('\n\ncross_val_index: ' + cross_val_index + '\n\n')
        if True:
            main(cross_val_index,args)
    log.write('The average accuracy is:',accuracy_total/5)
