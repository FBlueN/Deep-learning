import torch
import numpy
from torchvision import transforms,datasets
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
import os
import glob
import shutil
import csv


#############################################################################我是分隔線#############################################################################
# 更改係數區
device = torch.device('cpu') # 'cuda'/'cpu'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 設定當可以使用顯卡時就使用顯卡
batch_size = 64            # 每N張影像修正一次(沒抓到最好之前通常用128)，記憶體不夠的話要下降
learning_rate = 0.00001       # 修正幅度不能太大
epochs = 200                # 從1到最後跑過一遍叫做一個epochs # 有人會用evaluation 代表N筆資料除上batch_size，代表修正幾次
base_path = 'D:\orchid\pytorch\orchid' # 檔案位置
cut_num = 0.9              # 分割資料
final_output = int(219)           # 最終輸出(類別總數len(classes))
How_many_data = int(9000000000)  # 多少代後降低learning rate
USE_MULTI_GPU = True # 是否使用多顯卡
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218']  # 所有的類別
#############################################################################我是分隔線#############################################################################
# 資料分割
for cl in classes:
    img_path = os.path.join(base_path, cl)                          # 取得單一類別資料夾路徑
    images = glob.glob(img_path + '/*.jpg')                        # 載入所有 jpg 檔成為一個 list
    print("{}: {} Images".format(cl, len(images)))                 # 印出單一類別有幾張圖片
    num_train = int(round(len(images) * cut_num))                        # 切割 80% 資料作為訓練集
    train, val = images[:num_train], images[num_train:]            # 訓練 > 0~80%，驗證 > 80%~100%
    for t in train:
        if not os.path.exists(os.path.join(base_path, 'My_train', cl)):  # 如果資料夾不存在
            os.makedirs(os.path.join(base_path, 'My_train', cl))           # 建立新資料夾
        shutil.move(t, os.path.join(base_path, 'My_train', cl))          # 搬運圖片資料到新的資料夾

    for v in val:
        if not os.path.exists(os.path.join(base_path, 'My_valid', cl)):    # 如果資料夾不存在
            os.makedirs(os.path.join(base_path, 'My_valid', cl))             # 建立新資料夾
        shutil.move(v, os.path.join(base_path, 'My_valid', cl))            # 搬運圖片資料到新的資料夾
train_path = os.path.join(base_path, 'My_train')
valid_path = os.path.join(base_path, 'My_valid')
print("#############################################################################我是分隔線#############################################################################")
#############################################################################我是分隔線#############################################################################
# 資料增強
original_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 最基本的resize成可以用的圖以及把RGB歸一化

train_transforms_Color2 = transforms.Compose([transforms.ColorJitter(brightness = 0.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 改亮度0.5

train_transforms_Color3 = transforms.Compose([transforms.ColorJitter(contrast = 0.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 改對比度0.5

train_transforms_Color = transforms.Compose([transforms.ColorJitter(saturation = 0.3), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 改飽和度0.3                                   

train_transforms_Color4 = transforms.Compose([transforms.ColorJitter(hue = 0.3), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 改色調0.3

train_transforms_Color5 = transforms.Compose([transforms.ColorJitter(brightness = 0.3), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 改亮度0.3                                     

train_transforms_Vertical_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 垂直翻轉

train_transforms_Double_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1), transforms.RandomHorizontalFlip(p = 1), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 水平+垂直翻轉

train_transforms_Horizontal_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 水平翻轉

train_transforms_Grayscale = transforms.Compose([transforms.Grayscale(num_output_channels = 3), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 改成黑白圖

train_transforms_RandomAffine = transforms.Compose([transforms.RandomAffine(degrees = (-30,30), translate = (0, 0.5)), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 仿射變換

train_transforms_Rotation30 = transforms.Compose([transforms.RandomRotation(30, resample=None, expand=False, center = None, fill = None), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 旋轉30°

train_transforms_Rotation60 = transforms.Compose([transforms.RandomRotation(60, resample=None, expand = False, center = None, fill = None), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 旋轉60°

train_transforms_Pad = transforms.Compose([transforms.Pad((10, 5, 40, 20), fill = 0, padding_mode = "constant")]) # padding
"""
    1、最原始的 train:100、valid:40
    2、改亮度0.5 ~ 1.5 train:100、valid:44
    3、改對比度0.5 ~ 1.5 train:100、valid:43
    4、改飽和度0.7 ~ 1.3 train:不一定可以到100、valid:35 ~ 45浮動太大
    5、改色調0.7 ~ 1.3 train:有上100、valid:大概30 ~ 40
    6、改亮度0.7 ~ 1.3 train:100、valid:41
    7、垂直翻轉 train:100、valid:44
    8、垂直水平翻轉 train:100、valid:42
    9、水平翻轉 train:100、valid:46
    10、改灰階圖 train:100、valid:40
    11、仿射變換(後面參數很多可以用)
        1、+-30°的圖片旋轉，和會有水平以及垂直平移
    13、旋轉+-30°
    14、旋轉+-60°
    15、填充(10, 5, 40, 20)，填充黑色亦可用其他顏色或者讓padding的地方被拉伸或者鏡像
    16、以圖的中心切
    17、圖中任意位置切
    18、隨機裁切任意大小並且resize
    19、高斯模糊化
    20、隨機透視變換
    21、合成圖片
"""

#############################################################################我是分隔線#############################################################################
# 將資料載入並且豐富化(存在記憶體上)
train_data = datasets.ImageFolder(train_path, transform = original_transforms)
valid_data = datasets.ImageFolder(valid_path, transform = original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Color and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Color2 and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Color3 and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Color4 and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Color5 and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Vertical_flip and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Horizontal_flip and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Grayscale and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Pad and original_transforms)
train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_RandomAffine)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Double_flip and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Rotation30 and original_transforms)
# train_data = train_data + datasets.ImageFolder(train_path, transform = train_transforms_Rotation60 and original_transforms)
train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
valid_loader = data.DataLoader(valid_data, batch_size = batch_size, shuffle = True)
#############################################################################我是分隔線#############################################################################
# VGG16的模型
class VGG16_Model(nn.Module): # CNN_Model繼承nn.Module
    def __init__(self):
        super(VGG16_Model , self).__init__() # 等價nn.Module.__init__(self)
        self.vgg16 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), # batch_size*64*224*224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size  = 3, stride = 1, padding = 1), # batch_size*64*224*224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # batch_size*64*112*112

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), # batch_size*128*112*112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1), # batch_size*128*112*112
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # batch_size*128*56*56

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), # batch_size*256*56*56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), # batch_size*256*56*56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1), # batch_size*256*56*56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2), # batch_size*256*28*28

            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # batch_size*512*28*28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # batch_size*512*28*28 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # batch_size*512*28*28
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2 , stride = 2), # batch_size*512*14*14

            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # batch_size*512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # batch_size*512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1), # batch_size*512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # batch_size*512*7*7
        )
        self.classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, final_output)
        )
    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#############################################################################我是分隔線#############################################################################
if USE_MULTI_GPU and torch.cuda.device_count() > 1: # 判斷是否為多顯卡
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # 將顯卡作編號
    device_ids = [0, 1] # 設定GPU0為device_ids[0]GPU1為device_ids[1],預設卡0會作為reducer,卡1為加速用
else:
    MULTI_GPU = False
CNN = VGG16_Model()
if MULTI_GPU:
    CNN = nn.DataParallel(CNN , device_ids = device_ids , output_device = device_ids[0])
CNN.to(device)
#############################################################################我是分隔線#############################################################################
# 分類以及回歸
criterion = nn.CrossEntropyLoss() # 分類 包含softmax還有做log等等
#criterion=nn.MSELoss() # 回歸
optimizer = torch.optim.Adam(CNN.parameters(), lr = learning_rate) # 梯度的方法之一Adam
# optimizer = torch.optim.SGD(CNN.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, How_many_data, 0.1)  # 每N代就讓learning乘以0.1(讓整個圖開始收斂)
#############################################################################我是分隔線#############################################################################
# 找繪圖的參數
train_acc_his, train_losses_his, valid_acc_his, valid_losses_his = [], [], [], []
for i in range(0, epochs):
    print('running epoch:' + str(i + 1))
    train_correct, train_loss, valid_correct, valid_loss = 0, 0, 0, 0
    CNN.train()
    for image, label in train_loader: # 一個batch的image、label。image：(batch_size*3*3*224)。label：(1*batch_size)
        image, label = image.to(device), label.to(device)
        pred = CNN(image) # pred：(batch_size*class)
        loss = criterion(pred, label) # loss.data：(1*1)，一個batch的平均loss
        output_id = torch.max(pred, dim = 1)[1] # output_id：(1*batch_size)的網路輸出編號(0表示預測為第一個輸出)
        train_correct += numpy.sum(torch.eq(label, output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數
        train_loss += loss.item() * image.size(0) # 累加計算每一epoch的loss總和。loss.item()：(1*1)，一個batch的平均loss。image.size(0)：一個batch的訓練資料總數
        optimizer.zero_grad() # 權重梯度歸零
        loss.backward() # 計算每個權重的loss梯度
        optimizer.step() # 權重更新
    scheduler.step() # 降低lr
    CNN.eval()
    for image, label in valid_loader: # 一個batch的image、label。image：(batch_size*3*3*224)。label：(1*batch_size)
        image, label = image.to(device), label.to(device)
        pred = CNN(image) # pred：(batch_size*class)
        loss = criterion(pred, label) # loss.data：(1*1)，一個batch的平均loss
        output_id = torch.max(pred, dim = 1)[1] # output_id：(1*batch_size)的網路輸出編號(0表示預測為第一個輸出)
        valid_correct += numpy.sum(torch.eq(label, output_id).cpu().numpy()) # 累加計算每一epoch正確預測總數
        valid_loss += loss.item() * image.size(0) # 累加計算每一epoch的loss總和。loss.item()：(1*1)，一個batch的平均loss。image.size(0)：一個batch的驗證資料總數
    # print(train_total)
    train_acc = train_correct / len(train_loader.dataset) * 100 # 計算每一個epoch的平均訓練正確率(%)
    train_loss = train_loss / len(train_loader.dataset) # 計算每一個epoch的平均訓練loss
    valid_acc = valid_correct / len(valid_loader.dataset) * 100 # 計算每一個epoch的平均驗證正確率(%)
    valid_loss = valid_loss / len(valid_loader.dataset) # 計算每一個epoch的平均驗證loss
    train_acc_his.append(train_acc) # 累積紀錄每一個epoch的平均訓練正確率(%) (1*epochs)
    train_losses_his.append(train_loss) # 累積記錄每一個epoch的平均訓練loss (1*epochs) 
    valid_acc_his.append(valid_acc) # 累積紀錄每一個epoch的平均驗證正確率(%) (1*epochs)
    valid_losses_his.append(valid_loss) # 累積記錄每一個epoch的平均驗證loss (1*epochs)
    # 將每一代的訓練驗證的loss以及正確率都print出來
    print('train loss:', train_loss)
    print('train acc:', train_acc)
    print('valid loss:', valid_loss)
    print('valid acc:', valid_acc , '\n')
#############################################################################我是分隔線#############################################################################

# 繪圖用
plt.figure(figsize = (15 , 10))
plt.subplot(211)
plt.plot(train_acc_his, 'b', label = 'training accuracy')
plt.plot(valid_acc_his, 'r', label = 'validation accuracy')
plt.title('Accuracy(%)')
plt.legend(loc = 'best')
plt.subplot(212)
plt.plot(train_losses_his, 'b', label = 'training loss')
plt.plot(valid_losses_his, 'r', label = 'validation loss')
plt.title('Loss')
plt.legend(loc = 'best')
plt.show()

# 存模型以及之後可以載入模型作運用
torch.save(CNN, "module1")
best = torch.load('module')