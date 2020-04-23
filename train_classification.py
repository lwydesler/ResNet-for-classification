import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from resnet_classification import ResNet18

img_data = torchvision.datasets.ImageFolder(r'E:/geoclass/class/',
                                            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                          transforms.RandomVerticalFlip(),
                                                                          transforms.ColorJitter(contrast=1),
                                                                          transforms.ColorJitter(brightness=1),
                                                                          transforms.ToTensor()]))
img_data.class_to_idx

train_size = int(0.8 * len(img_data))
test_size = len(img_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(img_data, [train_size, test_size])
print(len(train_dataset))
print(len(test_dataset))

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size)
print(len(train_loader))
print(len(test_loader))

model = ResNet18()
model.cuda()
print(model)

    # 损失函数
criterion = torch.nn.CrossEntropyLoss()
    # 优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

accuracy = []
loss_line = []
start_time = time.time()
num_epochs = 500
times = 0
train_acc = 0
for epoch in range(num_epochs):

        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            preds = model(x)

            k, l = torch.max(preds, 1)
            num_acc = l
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            train_correct = num_acc.eq(y.data).cpu().sum()
            acc = train_correct.item() / batch_size

        print('epoch', epoch + 1, '批次', times + 1, 'loss', loss.data.cpu().numpy(), 'acc:', acc)
        loss_line.append(loss)
        accuracy.append(acc)

finish_time = time.time()
print(finish_time - start_time)
torch.save(model, 'E:/geoclass/model/model_217.pth')

x_crod2 = []
x_crod = len(accuracy)
for x_value in range(0, x_crod):
    x_crod2.append(x_value)

plt.figure(figsize=(8, 4)) #创建绘图对象
plt.plot(x_crod2, loss_line, "g", linewidth=0.5)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("epoic(s)") #X轴标签
plt.ylabel("loss_value")  #Y轴标签
plt.title("loss value plot") #图标题


plt.figure(figsize=(8, 4)) #创建绘图对象
plt.plot(x_crod2, accuracy, "r", linewidth=0.5)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("epoic(s)") #X轴标签
plt.ylabel("accuracy_value")  #Y轴标签
plt.title("accuracy") #图标题
plt.show()