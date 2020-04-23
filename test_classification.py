import torchvision
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import tqdm

transform = torchvision.transforms.Compose([transforms.ToTensor()])


def read_value(txt_adress):
    class_list = []
    with open(txt_adress, 'r') as file:
        for line in file:
            print(line)
            class_list.append(int(line))
        print(class_list[-1])
    file.close()
    return class_list

def prediect(img_path):
    net = torch.load('E:/geoclass/model/model_217.pth')
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


def write_predict_value(txt_adress):
    lenth = 9030
    #9030
    test_num = []
    i = 1
    while i <= lenth:
        img_path = 'E:/geoclass/test/test2/' + str(i) + '.png'
        predict_value = prediect(img_path)
        test_num.append(predict_value[0])
        i = i + 1
        print(i)
    with open(txt_adress, 'w') as file:
        for i in test_num:
            i1 = int(i)
            print(type(i1))
            file.write(str(i1)+'\n')
        file.close()

def score(test_txt, real_txt):
    test_num = read_value(test_txt)
    test_num1 = np.array(test_num)
    real_num = read_value(real_txt)
    real_num1 = np.array(real_num)
    p = precision_score(test_num1, real_num1, average=None)
    r = recall_score(test_num1, real_num1, average=None)
    f1score = f1_score(test_num1, real_num1, average=None)
    print('precision:' + str(p))
    print('recall:' + str(r))
    print('f1score:' + str(f1score))

if __name__ == "__main__":
    test_adress = 'E:/geoclass/test_value_part2.txt'
    #reall_adress = 'E:/geoclass/real_value_part1.txt'
    #write_predict_value(test_adress)

    score_test_adress = 'E:/geoclass/test_value_all.txt'
    score_real_adress = 'E:/geoclass/real_value_all.txt'

    score(score_test_adress, score_real_adress)