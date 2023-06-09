import torch
import torchvision
import os
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.distributions import transforms
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataSet
from nets.TICNN import *
from nets.TICNN_2048 import *
import time
from utils import read_split_data

#tensorboard使用方法：tensorboard --logdir "E:\Python\Fault Diagnosis\Classification\logs"
#需要设置cuda的数据有: 数据，模型，损失函数

save_epoch=1 #模型保存迭代次数间隔-10次保存一次

#定义训练的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 训练参数
batch_size = 10
epoch = 100

learing_rate=1e-3 #学习速率
#准备数据集
#加载自制数据集

root_source = ".\dataset/1024/A"  # 数据集所在根目录-【资源域】
root_target = ".\dataset/1024/D_SNR4"  # 数据集所在根目录-【目标域】



#读取资源域数据
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root_source,0.005)
#读取目标域数据
target_train_images_path, target_train_images_label, target_val_images_path, target_val_images_label = read_split_data(root_target,0.998)

train_data_set = MyDataSet(images_path=train_images_path,
                           images_class=train_images_label,
                           transform="1")
test_data_set = MyDataSet(images_path=val_images_path,
                          images_class=val_images_label,
                          transform="1")
tgt_train_data_set = MyDataSet(images_path=target_train_images_path,
                               images_class=target_train_images_label,
                               transform="1")
tgt_test_data_set = MyDataSet(images_path=target_val_images_path,
                              images_class=target_val_images_label,
                              transform="1")


train_data_size=len(train_data_set)
test_data_size=len(test_data_set)
tgt_train_data_size = len(tgt_train_data_set)  # 目标域数据集长度
tgt_test_data_size = len(tgt_test_data_set)  # 目标域数据集长度
print('训练样本数', train_data_size)
print('测试样本数', tgt_test_data_size)

#加载数据集
train_dataloader = torch.utils.data.DataLoader(train_data_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=train_data_set.collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=test_data_set.collate_fn)

tgt_train_dataloader = torch.utils.data.DataLoader(tgt_train_data_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   collate_fn=tgt_train_data_set.collate_fn)
tgt_test_dataloader = torch.utils.data.DataLoader(tgt_test_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=tgt_test_data_set.collate_fn)

wang=TICNN().to(device)  # 将模型加载到cuda上训练
# wang=TICNN_2048().to(device)  # 将模型加载到cuda上训练

#定义损失函数
loss_fn=nn.CrossEntropyLoss().to(device) #将损失函数加载到cuda上训练

#定义优化器

#optimizer=torch.optim.SGD(wang.parameters(),lr=learing_rate)softmax
optimizer = Adam(wang.parameters(), lr=learing_rate)  # 选用AdamOptimizer
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #使用学习率指数连续衰减

#设置训练网络的一些参数
total_train_step=0 #记录训练的次数
total_test_step=0 #记录测试的次数

#添加tensorboard
# writer=SummaryWriter("logs",flush_secs=5)

test_accuracy = np.array([])
target_test_accuracy = np.array([])
model_name = np.array([])
for i in range(epoch):
    print("---------第{}轮训练开始------------".format(i+1))
    total_train_loss = 0  # 训练集整体Loss
    #训练步骤开始
    wang.train() #会对归一化及dropout等有作用
    for data in train_dataloader:
        imgs, targets=data #取图片数据
        #imgs=torch.squeeze(imgs,dim=3)
        #targets = torch.tensor(targets, dtype=torch.long)
        imgs = imgs.type(torch.cuda.FloatTensor)
        imgs = imgs.to(device)  # 将图片加载到cuda上训练
        targets = targets.to(device)  # 加载到cuda上训练
        outputs=wang(imgs) #放入网络训练
        loss=loss_fn(outputs,targets) #用损失函数计算误差值
        #优化器调优
        optimizer.zero_grad() #清零梯度
        loss.backward() #反向传播
        optimizer.step()

        total_train_loss = total_train_loss + loss.item()
        total_train_step=total_train_step+1
        if total_train_step%10==0:
            print("总训练次数: {},损失值Loss: {}".format(total_train_step,loss.item()))
            # writer.add_scalar("train_loss",loss.item(),global_step=total_train_step)
    if i % 2 == 0:
        scheduler.step()
    current_learn_rate=optimizer.state_dict()['param_groups'][0]['lr']
    print("当前学习率：", current_learn_rate, "---------------------------")
    print("第{}训练后的【训练集-整体】Loss为: {}".format(i + 1, total_train_loss))

    #一轮训练后，进行测试
    wang.eval()
    total_test_loss=0 #总体loss
    total_correct_num=0 #总体的正确率
    transfer_total_test_loss = 0  # 总体loss
    transfer_total_correct_num = 0  # 总体的正确个数
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets=data
            #imgs = torch.squeeze(imgs, dim=3)
            imgs = imgs.type(torch.cuda.FloatTensor)
            #targets = torch.tensor(targets, dtype=torch.long)
            imgs = imgs.to(device)  # 将图片加载到cuda上训练
            targets = targets.to(device)  # 加载到cuda上训练
            outputs=wang(imgs)
            loss=loss_fn(outputs,targets) #单个数据的loss
            total_test_loss=total_test_loss+loss+loss.item()
            correct_num=(outputs.argmax(1)==targets).sum() #1:表示横向取最大值所在项
            total_correct_num=total_correct_num+correct_num #计算预测正确的总数
    test_accuracy = np.append(test_accuracy, (total_correct_num / test_data_size).cpu())  # 保存每次迭代的测试准确率
    print("第{}训练后的【测试集-资源域】总体Loss为: {}".format(i + 1, total_test_loss))
    print("第{}训练后的【测试集-资源域】总体正确率为: {}".format(i + 1, total_correct_num / test_data_size))
    # writer.add_scalar("test_loss",total_test_loss, total_test_step) #添加测试loss到tensorboard中
    # writer.add_scalar("test_accuracy",total_correct_num/test_data_size,total_test_step) #添加测试数据集准确率到tensorboard中
    total_test_step=total_test_step+1

    # --------------------跨域测试----------------------#
    with torch.no_grad():
        for data in tgt_test_dataloader:
            imgs, targets = data
            imgs = imgs.type(torch.cuda.FloatTensor)
            imgs = imgs.to(device)  # 将图片加载到cuda上训练
            targets = targets.to(device)  # 加载到cuda上训练
            outputs = wang(imgs)
            loss_transfer = loss_fn(outputs, targets)  # 单个数据的loss
            transfer_total_test_loss = transfer_total_test_loss + loss_transfer
            correct_num = (outputs.argmax(1) == targets).sum()  # 1:表示横向取最大值所在项
            transfer_total_correct_num = transfer_total_correct_num + correct_num  # 计算预测正确的总数
    target_test_accuracy = np.append(target_test_accuracy,
                                     (transfer_total_correct_num / tgt_test_data_size).cpu())  # 保存每次迭代的测试准确率
    print("第{}训练后的【测试集-目标域】总体Loss为: {}".format(i + 1, transfer_total_test_loss))
    print("第{}训练后的【测试集-目标域】总体正确率为: {}".format(i + 1, transfer_total_correct_num / tgt_test_data_size))

    time_str=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_path = './models/'
    filepath = os.path.join(save_path, "wang_{}_{}.pth".format(time_str,i+1))
    model_name = np.append(model_name, filepath)
    if (i+1) % save_epoch == 0:
        torch.save(wang,filepath) #保存训练好的模型

index=np.argmax(target_test_accuracy)
str2='第{}次迭代【测试集-资源域】准确率:{}'.format(np.argmax(target_test_accuracy)+1,test_accuracy[index])
str3='第{}次迭代【测试集-目标域】最大准确率:{}'.format(np.argmax(target_test_accuracy) + 1, np.max(target_test_accuracy))
str4='第{}次迭代对应的模型名称:{}'.format(np.argmax(target_test_accuracy) + 1,model_name[index])
print(str2)
print(str3)
print(str4)

with open('./logs/result.txt', 'a') as file:
    file.write(str2 + '\n')
    file.write(str3 + '\n')
    file.write(str4 + '\n')

# writer.close() #关闭tensorboard

