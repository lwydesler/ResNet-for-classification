import torchvision
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

transform = torchvision.transforms.Compose([transforms.ToTensor()])


def read_value(txt_adress):
    class_list = []
    with open(txt_adress, 'r') as file:
        for line in file:
            for class_num in line.split(' '):
                print(class_num)
                class_list.append(class_num)
            print(class_list[-1])
    file.close()
    return class_list

def prediect(img_path):

    net = torch.load('E:/geoclass/model/model_216.pth')
    net = net.cuda()
    net.eval()
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.cuda()
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    x = predicted.cpu().numpy()

    return x

def write_predict_value(txt_adress, number):
    test_num = []
    i = 1
    while i <= number:
        img_path = 'E:/geoclass/predict_25/' + str(i) + '.png'
        predict_value = prediect(img_path)
        test_num.append(predict_value)
        print(i)
        i = i + 1
    with open(txt_adress, 'w') as file:
        for i in test_num:
            k = int(i)
            file.write(str(k) + '\n ')
    file.close()

if __name__ == "__main__":
    test_adress = 'E:/geoclass/predict_value_25.txt'
    write_predict_value(test_adress, 82521)