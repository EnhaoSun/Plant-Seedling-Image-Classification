import os
import time
import shutil
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import ImageFolder
import argparse
#from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--model_name", help="specify model name to save")
parser.add_argument("--size", help="specify image size")
args = parser.parse_args()
if_gpu = torch.cuda.is_available()
print("GPU is on?", if_gpu)


# Define a dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        return self.to_tensor(self.x_data[index]), self.y_data[index]

    def __len__(self):
        return len(self.x_data)

# Note that torchvision.transforms.ToTensor() will
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range
# [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

size = args.size
train_dict = np.load("data/" + str(size) + "/plant-train-data.npz")
whole_dataset = ImageDataset(train_dict["data"], train_dict["labels"])

print(whole_dataset[0][0].shape)
#print(whole_dataset[4610])

# subset the whole train set for accuracy check while training.
def array_random_pick(array, pick_num):
    index = np.arange(len(array))
    pick = np.random.choice(len(array), pick_num, replace=False)
    unpick = np.equal(np.in1d(index, pick), False)
    return array[unpick], array[pick]

train_mask, valid_mask = array_random_pick(np.arange(len(whole_dataset)), 500)

train_set = torch.utils.data.Subset(whole_dataset, train_mask)
valid_set = torch.utils.data.Subset(whole_dataset, valid_mask)

print(len(train_set),len(valid_set))
#print(train_set[4010])
#print(valid_set[401])

# Use DataLoader to group data batchs. Here use size 4 for a batch.
# DataLoader will return a iterator.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=5)
load_iter = iter(train_loader)
one_batch_x, one_batch_y = next(load_iter)

#print(one_batch_y)
#print(one_batch_x.shape)


# Use PyTorch's built-in model to generate AlexNet with classes 12.
# With input data of size [4, 3, 224, 224], AlexNet will output data of size [4, 12].

# if (args.model_name == 'AlexNet'):
#     model = torchvision.models.AlexNet(num_classes = 12)
#     out = model(one_batch_x)
# elif (args.model_name == 'Inception3'):
#     pass
#     # model = torchvision.models.Inception3(num_classes = 12)
#     # out = model(one_batch_x)
# else:
#     print
#     exit(1)

#print(out.shape)
#print(out)


# We use the max index of out to
# evaluate the accuracy of model predict.
# Now the accuracy is zero before model train.
#predict = torch.argmax(out, dim = 1)
#compare = predict == one_batch_y
#accuracy = compare.sum() / len(predict)

'''
print(predict)
print(one_batch_y)
print(compare)
print("accuracy =", accuracy.data.numpy())
'''

# Print model's state_dict


# define a base utility to train a net.
class BaseNetPyTorch:
    def __init__(self):
        self.train_loader = None
        self.sub_train_loader = None
        self.valid_loader = None

        self.model = None
        self.optimize_method = None
        self.loss_function = None

        if_gpu = torch.cuda.is_available()
        self.device_gpu = torch.device("cuda:0" if if_gpu else "cpu")

    def train_loss(self):
        # "training" mode for Dropout etc.
        self.model.train()

        train_loss = None
        for (x, y) in self.train_loader:
            x_gpu = x.to(self.device_gpu)
            y_gpu = y.long().to(self.device_gpu)

            predict = self.model(x_gpu)
            train_loss = self.loss_function(predict, y_gpu)

            self.optimize_method.zero_grad()
            train_loss.backward()
            self.optimize_method.step()
        return train_loss

    def predict_index(self, check_loader):
        predict_list = []
        for x, y in check_loader:
            x_gpu = x.to(self.device_gpu)
            predict = self.model(x_gpu)
            max_index = torch.argmax(predict, dim=1)
            predict_list += max_index.cpu().data.numpy().tolist()
        return predict_list

    def check_accuracy(self, check_set):
        num_correct = 0
        num_samples = 0
        # "test" mode for Dropout etc.
        self.model.eval()
        for x, y in check_set:
            x_gpu = x.to(self.device_gpu)
            y_gpu = y.to(self.device_gpu)

            predict = self.model(x_gpu)
            max_index = torch.argmax(predict, dim=1)

            num_correct += (max_index == y_gpu).sum()
            num_samples += max_index.shape[0]

        accuracy = float(num_correct) / float(num_samples)
        return num_correct, num_samples, accuracy

    def train(self, num_epochs=1):
        if self.model is None:
            raise ValueError("self.model is None! Please assign it.")
        if self.optimize_method is None:
            raise ValueError("self.optimize_method is None! Please assign it.")
        if self.loss_function is None:
            raise ValueError("self.loss_function is None! Please assign it.")

        print("begin training, length_of_one_mini_batch :", len(self.train_loader))

        self.model = self.model.to(self.device_gpu)

        train_time = time.time()
        for epoch in range(num_epochs):
            epoch_start = time.time()

            loss = self.train_loss()
            loss_time = time.time()

            train_correct, train_samples, train_acc = self.check_accuracy(self.sub_train_loader)
            train_acc_time = time.time()

            valid_correct, valid_samples, valid_acc = self.check_accuracy(self.valid_loader)
            valid_acc_time = time.time()

            epoch_time = time.time()

            print('epoch:%d/%d' % (epoch + 1, num_epochs), end=" ")
            print('loss:%.4f|%ds' % (loss.data, (loss_time - epoch_start)), end=" ")
            print('train_acc:(%d/%d %0.2f%%)|%ds' %
                  (train_correct, train_samples, 100 * train_acc, train_acc_time - loss_time), end=' ')
            print('valid_acc:(%d/%d %0.2f%%)|%ds' %
                  (valid_correct, valid_samples, 100 * valid_acc, (valid_acc_time - train_acc_time)), end=' ')
            print("take:%dmin remain:%dmin" %
                  ((epoch_time - train_time) / 60, (epoch_time - epoch_start) * (num_epochs - epoch) / 60))

            if (train_acc - 0.3 > valid_acc) and (train_acc > 0.5):
                print("Model Overfit 30.00%, stopped.")
                return True

        return True


# It is very important to turn on shuffle=True of training set
train_loader = torch.utils.data.DataLoader(train_set, batch_size=40, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=40)

net = BaseNetPyTorch()
if (args.model_name == 'AlexNet'):
    net.model = torchvision.models.AlexNet(num_classes=12)
elif (args.model_name == 'Inception3'):
    net.model = torchvision.models.Inception3(num_classes=12)
else:
    print("specify --model_name")
    exit(1)

net.optimize_method = torch.optim.Adam(net.model.parameters(), lr=0.0001)
net.loss_function = torch.nn.CrossEntropyLoss()
net.train_loader = train_loader
net.sub_train_loader = train_loader
net.valid_loader = valid_loader

#save state at prespecified filepath
model_save_dir = "models"
model_idx = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
f_model=os.path.join(model_save_dir, "{}_{}".format(args.model_name, str(model_idx)))
f_prediction=os.path.join(model_save_dir, "{}_{}".format("prediction", str(model_idx))) + ".csv"

print(f_model);
net.train(30)
torch.save(net.model.state_dict(), f_model)

# predict test file labels
test_dict = np.load("data/" + str(size) + "/plant-test-data.npz")
test_set = ImageDataset(test_dict["data"], test_dict["labels"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=40)
test_predict = net.predict_index(test_loader)

#label_names = train_info_dict["label_names"]
#label_names = np.reshape(label_names,1)
#predict_names = [label_names[0].get(i) for i in test_predict]

print(test_set.y_data[:10])
print(test_predict[:10])

y_true = test_set.y_data
y_pred = test_predict

accuracy = accuracy_score(y_true, y_pred)
accuracy = np.reshape(accuracy,1)
print(accuracy)


df = pd.DataFrame(accuracy, columns=['test_acc'])
df.to_csv(f_prediction, index=False)
'''
'''
