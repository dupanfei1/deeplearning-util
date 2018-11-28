# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from dataset.dataset import dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
test_transforms= transforms.Compose([
            transforms.RandomRotation(degrees=22),
            transforms.RandomResizedCrop(224, scale=(0.49, 1.0)),
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

rawdata_root = '/home/lab548/Desktop/bd_tiangong/dataset'
test_pd = pd.read_csv("/home/lab548/Desktop/bd_tiangong/dataset/test7.txt",sep=" ",
                       header=None, names=['ImageName'])
test_pd['label'] =1

data_set = {}
data_set['test'] = dataset(imgroot=os.path.join(rawdata_root, 'test'), anno_pd=test_pd,
                             transforms=test_transforms,
                             )
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=7, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

model_name = 'densenet'
resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-18-1006-[0.9961].pth'#97.7
#resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-17-1039-[0.9968].pth'#0.974
#resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-15-705-[0.9958].pth'#96.9
#resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-14-2338-[0.9958].pth'#97.5
#resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-14-1138-[0.9958].pth'#97.5
# resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-14-738-[0.9944].pth'#97.3
# resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-13-2771-[0.9961].pth'#97.2
# resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-14-1938-[0.9951].pth'#97.8
# resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense169_bigdata_train/weights-15-305-[0.9958].pth'#97.5
# resume = '/home/lab548/Desktop/bd_tiangong/trainedmodel/dense1_bigdata_train/weights-6-202-[0.9887].pth'

model=models.densenet161(pretrained=True)
# model.fc = torch.nn.Linear(1024,100)
# model.fc=F.log_softmax(torch.nn.Linear(1000, 10))

# model = make_model('inceptionresnetv2', num_classes=1000, pretrained=True, pool=nn.AdaptiveMaxPool2d(1))
# print(model.original_model_info)
# model = make_model('se_resnet152', num_classes=1000, pretrained=True, input_size=(224, 224) ,pool=nn.AdaptiveMaxPool2d(1))

model.fc = torch.nn.Sequential(#对应修改
    torch.nn.Linear(1000, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 100)
)
# model.classifier = torch.nn.Sequential(#看最后一层的 名字是啥,修改输出维度
#     torch.nn.Linear(2208, 512),
#     torch.nn.ReLU(),
#     torch.nn.Linear(512, 6)
# )

# print model

model.load_state_dict(torch.load(resume))
model = model.cuda()
model.eval()
criterion = CrossEntropyLoss()

if not os.path.exists('./csv'):
    os.makedirs('./csv')

print len(data_set['test'])
test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0
result= []
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data

    print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    outputs = model(inputs)
    out2 = np.array(outputs.data)
    outsum = out2.sum(axis = 0)

    out_soft = F.softmax(outputs, dim=1)#使用softmax
    out_soft2 = np.array(out_soft.data)
    out_softsum = out_soft2.sum(axis = 0)

    result.append(np.argmax(outsum))
    # statistics
    if isinstance(outputs, list):
        loss = criterion(outputs[0], labels)
        loss += criterion(outputs[1], labels)
        outputs = (outputs[0]+outputs[1])/2
    else:
        loss = criterion(outputs, labels)
    _, preds = torch.max(outputs, 1)


    test_loss += loss.data[0]
    batch_corrects = torch.sum((preds == labels)).data[0]
    test_corrects += batch_corrects
    test_preds[idx:(idx + labels.size(0))] = preds
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
    # statistics
    idx += labels.size(0)
test_loss = test_loss / test_size
test_acc = 1.0 * test_corrects / len(data_set['test'])

label2num = {1:'OCEAN',2:'DESERT',3:'MOUNTAIN',4:'FARMLAND',5:'LAKE',6:'CITY'}




#写入综合结果
test_pd = pd.read_csv("/home/lab548/Desktop/bd_tiangong/dataset/test.txt",sep=" ",
                       header=None, names=['ImageName'])
test_pd['label'] =1
test_pred = test_pd[['ImageName']].copy()
# test_pred['label'] = list(test_preds)
print len(result)
test_pred['label'] = result
test_pred['label'] = test_pred['label'].apply(lambda x: label2num[int(x)+1])
test_pred[['ImageName',"label"]].to_csv('csv/{0}_{1}.csv'.format(model_name,'test') ,sep=",",
                                                                 header=None, index=False)
