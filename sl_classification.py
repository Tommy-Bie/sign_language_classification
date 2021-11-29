import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import Image, ImageEnhance
from shutil import copy
import cv2
import PIL
import argparse
import warnings
from my_CNN import ConvolutionalNetwork
from my_ALEXNET import alexnet
from my_VGG import vgg16
from my_GOOLENET import googlenet

# GPU选择
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 数据集划分与读取
def sl_data_split(split_root , dest_root):
    for letter in  os.listdir(os.path.join(split_root)):
        for cnt,img in enumerate(os.listdir(os.path.join(split_root,letter))):
            cont = False
            if letter =='0': new_gest = '0'; cont = True
            elif letter =='1': new_gest = '1'; cont = True
            elif letter =='2': new_gest = '2'; cont = True
            elif letter == '6': new_gest = '3'; cont = True
            elif letter =='4': new_gest = '4'; cont = True
            elif letter =='5': new_gest = '5'; cont = True

            if cont == True:
                img_str = letter + '-' + img
                if cnt/len(os.listdir(os.path.join(split_root,letter))) < 0.75:
                    if not os.path.exists(os.path.join(dest_root,'train',new_gest,img_str)):
                        copy(os.path.join(split_root,letter,img), os.path.join(dest_root,'train',new_gest,img_str))
                else:
                    if not os.path.exists(os.path.join(dest_root,'test',new_gest,img_str)):
                        copy(os.path.join(split_root,letter,img), os.path.join(dest_root,'test',new_gest,img_str))

# dataloader
def data_loader(root):
    train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)
    global train_loader
    global test_loader
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=True, pin_memory=True, num_workers=0)


# 训练与测试函数
def train_CNN(epochs, save_root, network):
    start_time = time.time()  # 记录训练时间
    max_trn_batch = 800  # Limits very large datasets
    max_tst_batch = 300
    train_losses = []  # 损失
    test_losses = []
    train_correct = []  # 正确
    test_correct = []

    for i in range(epochs):  # 训练
        trn_corr = 0
        tst_corr = 0
        for b, (X_train, y_train) in enumerate(train_loader): # batch训练
            if b == max_trn_batch: break  # 如果迭代的batch的个数超过800则结束训练
            b += 1
            #if torch.cuda.is_available(): # Turn tensors to cuda
            X_train = X_train.to(DEVICE, non_blocking=True)
            y_train = y_train.to(DEVICE, non_blocking=True)

            if network == 'myCNN' or 'alexnet' or 'vgg':
                y_pred = CNNmodel(X_train)  # CNN模型：myCNN/AlexNet/VGG16
            elif network == 'googlenet':
                y_pred, aux2, aux1 = CNNmodel(X_train)  # GoogLeNet模型
            loss = criterion(y_pred, y_train)  # 损失，使用的交叉熵损失

            predicted = torch.max(y_pred.data, 1)[1]  # 正确的预测个数合计
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr  # 加入总的正确预测个数

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()

            if b % 50 == 0:  # 每50个batch打印一次损失和正确率
                print(f'epoch: {i:2}  batch: {b:4} loss: {loss.item():10.8f} '
                      f'train accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')


        train_losses.append(loss)  # 训练损失记录
        train_correct.append(trn_corr)  #  预测正确数记录

        with torch.no_grad():  # batch测试
            for b, (X_test, y_test) in enumerate(test_loader):
                if b == max_tst_batch: break
                # if torch.cuda.is_available(): # Turn tensors to cuda
                X_test = X_test.to(DEVICE, non_blocking=True)
                y_test = y_test.to(DEVICE, non_blocking=True)

                if network == 'myCNN' or 'alexnet' or 'vgg':
                    y_val = CNNmodel(X_test)  # CNN模型：myCNN/AlexNet/VGG16
                elif network == 'googlenet':
                    y_val, aux2, aux1 = CNNmodel(X_test)  # GoogLeNet模型
                loss = criterion(y_val, y_test)

                predicted = torch.max(y_val.data, 1)[1]  # Tally the number of correct predictions
                tst_corr += (predicted == y_test).sum()

        test_losses.append(loss)
        test_correct.append(tst_corr)

    # 总测试
    my_tst_corr=0
    for a, (X_test, y_test) in enumerate(test_loader):

        X_test = X_test.to(DEVICE, non_blocking=True)
        y_test = y_test.to(DEVICE, non_blocking=True)

        if network == 'myCNN' or 'alexnet' or 'vgg':
            y_val = CNNmodel(X_test)  # CNN模型：myCNN/AlexNet/VGG16
        elif network == 'googlenet':
            y_val, aux2, aux1 = CNNmodel(X_test)  # GoogLeNet模型

        predicted = torch.max(y_val.data, 1)[1]  # Tally the number of correct predictions
        my_tst_corr += (predicted == y_test).sum()

    print(f'test accuracy:{my_tst_corr * 100 / 306:7.3f}%')  # 总测试正确率
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # 训练和测试结束时打印总时间

    # 绘制损失曲线
    plt.plot(train_losses[1:7], label='训练损失')
    plt.plot(test_losses[1:7], label='测试损失')
    plt.title('损失曲线 Loss')
    plt.legend()
    plt.show()
    # 绘制正确率曲线
    plt.plot([t/9.3 for t in train_correct], label='训练正确率')
    plt.plot([t/3. for t in test_correct], label='测试正确率')
    plt.title('正确率Accuracy')
    plt.legend()
    plt.show()

    torch.save(CNNmodel.state_dict(), save_root)  # 保存权重

def load_weights(weights_root):  # 权重读取
    if torch.cuda.is_available():
        CNNmodel.load_state_dict(torch.load(weights_root))
    else:
        CNNmodel.load_state_dict(torch.load(weights_root, map_location=torch.device('cpu')))
    CNNmodel.eval


# 测试单张图片
def test_image(test_img_root):
    im2 = Image.open(test_img_root).convert('RGB')
    enhancer = ImageEnhance.Contrast(im2)  # change contrast for better results
    im2 = enhancer.enhance(.5)
    im2 = test_transform(im2)
    print(im2.shape)
    cv2.imshow('img',np.transpose(im2.numpy(), (1, 2, 0)))
    cv2.waitKey(500)

    if torch.cuda.is_available():
        im2 = im2.to(DEVICE)  # CNN Model Prediction
    CNNmodel.eval()
    with torch.no_grad():
        preds = CNNmodel(im2.view(1,3,224,224))
        print(preds)
        new_pred = preds.argmax()
    print(f'Predicted value: {new_pred.item()} ')
    #print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')


# 摄像机实时检测
def cam_detect():
    cam = cv2.VideoCapture(0)
    CNNmodel.eval()

    roi_top = 100
    roi_bottom = 300
    roi_left = 10
    roi_right = 210

    if opt.switch_side:
        roi_left = 400
        roi_right = 600

    # Intialize a frame count
    num_frames = 0

    # 一直循环，按q退出
    while True:
        # 获取图像
        ret, frame = cam.read()
        # height = frame.shape[0]
        # width = frame.shape[1]
        frame = cv2.flip(frame, 1)  # 翻转（倒像）

        # 生成RoI
        frame_copy = frame.copy()
        roi_copy = frame.copy()
        roi_image = PIL.Image.fromarray(roi_copy, "RGB")
        enhancer = ImageEnhance.Contrast(roi_image)
        roi_image = enhancer.enhance(opt.contrast)
        enhancer = ImageEnhance.Brightness(roi_image)
        roi_image = enhancer.enhance(opt.brightness)
        roi_np = np.array(roi_image)
        roi_np = roi_np[:, :, ::-1].copy()

        roi = roi_np[roi_top:roi_bottom, roi_left:roi_right]
        if opt.show_roi:
            cv2.imshow("roi", roi)
        roi2 = PIL.Image.fromarray(roi, "RGB")

        # 在图像上标出RoI
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
        num_frames += 1

        roi2 = test_transform(roi2) # 测试图像变换
        if torch.cuda.is_available():
            roi2 = roi2.to(DEVICE)

        with torch.no_grad():
            preds = CNNmodel(roi2.view(1,3,224,224))
            new_pred = preds.argmax()

        # 显示预测结果
        cv2.putText(frame_copy, str(new_pred.item()), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # 显示手形分割
        cv2.imshow("Finger Count", frame_copy)

        if cv2.waitKey(1) == ord('q'):  # q to quit
            break

    # 退出相机结束
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 忽略warnings

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-weights', nargs='+', type=str, default='', help='model.pt path(s)')  # 权重读取
    parser.add_argument('--data-split', action='store_true', help='use to split data')  # 数据集划分
    parser.add_argument('--split-root', nargs='+', type=str, default='', help='2 entries (data_root data_dest)')
    parser.add_argument('--train-data',  type=str, default='gesture_numbers_ds', help='triggers training and declares data location')  # 训练数据
    parser.add_argument('--save-weights', type=str, default='default.pt', help='save results to *.pt')  # 保存权重
    parser.add_argument('--test-image', type=str, default='', help='test the CNN on single image')  # 测试单张图片
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')  # 迭代轮数
    parser.add_argument('--contrast', type=float, default=1, help='edit roi for better cnn recognition')  # 对比度调节
    parser.add_argument('--brightness', type=float, default=1, help='edit roi for better cnn recognition')  # 亮度调节
    parser.add_argument('--cam-detect', action='store_true', help='use model on default camera')  # 摄像机实时分类
    parser.add_argument('--switch-side', action='store_true', help='switch side for roi')  # RoI换边
    parser.add_argument('--show-roi', action='store_true', help='show roi after transformation')  # 在实时检测时显示RoI
    parser.add_argument('--network', type=str, default='myCNN', help='my_CNN/alexnet/vgg/googlenet')  # 选择分类用的卷积神经网络
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 学习率选择
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')  # 优化器选择

    opt = parser.parse_args()
    print(opt)

    # 图像变换处理与增强
    train_transform = transforms.Compose([  # 训练图像变换
            transforms.RandomHorizontalFlip(),  # 图像翻转
            transforms.RandomRotation(10),      # 图像随机旋转
            transforms.Resize((224,224)),       # 图像放缩
            transforms.ToTensor(),              # 转换为张量
            transforms.Normalize([0.485, 0.456, 0.406],     # 标准化
                                 [0.229, 0.224, 0.225]) ])

    test_transform = transforms.Compose([  # 测试图像变换
            transforms.Resize((224,224)),  # 图像放缩
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize([0.485, 0.456, 0.406],     # 标准化
                                 [0.229, 0.224, 0.225])])

    inv_normalize = transforms.Normalize( # 反标准化，用于看到标准化前图片
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225] )

    # 建立卷积神经网络模型
    torch.manual_seed(31)
    if opt.network == 'myCNN':
        CNNmodel = ConvolutionalNetwork()  # 自己的CNN
    elif opt.network == 'googlenet':
        CNNmodel = googlenet()  # GoogLeNet
    elif opt.network == 'alexnet':
        CNNmodel = alexnet()  # AlexNet
    elif opt.network == 'vgg':
        CNNmodel = vgg16()  # VGG16
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    lr = opt.lr
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=lr)  # 优化器：Adam 学习率设置
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(CNNmodel.parameters(), momentum=0.8, lr=lr)

    if torch.cuda.is_available():
        CNNmodel = CNNmodel.to(DEVICE)
    CNNmodel

    if opt.data_split: # 数据划分
         sl_data_split(opt.split_root[0], opt.split_root[1])

    if opt.train_data != '': # 训练模型
        data_loader(opt.train_data)
        train_CNN(opt.epochs, opt.save_weights, opt.network)

    if opt.train_data == '': # 如果没有选择训练数据则读取权重（测试）
        load_weights(opt.load_weights)

    if opt.test_image != '':  # 测试单张图片
        test_image(opt.test_image)

    if opt.cam_detect:  # 使用相机
        cam_detect()
