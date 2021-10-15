import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='~/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def cifar_imshow():
	# get some random training images
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

	print("finish imshow")
	print("###############################")

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 84)
		self.fc4 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

def train(EPOCHS=2, device="cpu"):
	"""
	device		"cpu" or "cuda:0"
	
	"""
	
	start = time.time()

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # used for training only, not testing
	print("device: ", device)  # Assuming that we are on a CUDA machine, this should print a CUDA device

	net = Net().to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	print("Start training")

	for epoch in range(EPOCHS):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data[0].to(device), data[1].to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	torch.save(net.state_dict(), PATH_checkpoint)

	print('Finished Training in %.2f sec'%(time.time()-start))
	print("###############################")	                 

def test_one_batch():
	print('Start test_one_batch')
	dataiter = iter(testloader)
	images, labels = dataiter.next()

	# print images
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

	net = Net()
	net.load_state_dict(torch.load(PATH_checkpoint))
	outputs = net(images)

	_, predicted = torch.max(outputs, 1)

	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
		                      for j in range(4)))
	print("finish test_one_batch")
	print("###############################")	                 
		                 
def test():    	
	net = Net()
	net.load_state_dict(torch.load(PATH_checkpoint))          
	correct = 0
	total = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			# calculate outputs by running images through the network
			outputs = net(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))
	     
	print("finish test")
	print("###############################")                            
		                      
def test_per_class():
	net = Net()
	net.load_state_dict(torch.load(PATH_checkpoint))  

	# prepare to count predictions for each class
	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}

	# again no gradients needed
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predictions = torch.max(outputs, 1)
			# collect the correct predictions for each class
			for label, prediction in zip(labels, predictions):
				if label == prediction:
					correct_pred[classes[label]] += 1
				total_pred[classes[label]] += 1


	# print accuracy for each class
	for classname, correct_count in correct_pred.items():
		accuracy = 100 * float(correct_count) / total_pred[classname]
		print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
													accuracy))
	
	print("finish test_per_class")
	print("###############################")                                    





#############################################
PATH_checkpoint = './cifar_net.pth'

# cifar_imshow()
train(EPOCHS=1, device="cpu")
train(EPOCHS=1, device="cuda:0")
prit("cuda should be ~x2 faster")
# test_one_batch()
# test()
# test_per_class()
