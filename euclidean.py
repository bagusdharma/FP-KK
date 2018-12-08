import csv
import random
import math
import operator
import numpy as np
import pandas


def euclidean_distance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		# print("instance1 :", instance1[x])
		# print("instance2 :", instance2[x])
		distance += pow((int(instance1[x]) - (instance2[x])), 2)
	return math.sqrt(distance)

def get_neighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclidean_distance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def get_response(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def get_accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


def splitting_data(dataset, split, trainset, test):
	new_data=[]
	for i in range(len(dataset)):
		sec_lev = []
		for y in range(1, 11):
			sec_lev.append(float(dataset[i][y]))
		new_data.append(sec_lev)
		if random.random() < split:
			trainset.append(new_data[i])
		else:
			test.append(new_data[i])


if __name__ == '__main__':
	trainingSet=[]
	testSet=[]
	split = 0.67
	predictions=[]

	# Processing the missing value
	dataset = pandas.read_csv("breastcancer.data", header=None)
	# dataset[[1,2,3,4,5,6,7,8,9,10]].replace('NaN', np.NaN)
	dataset.fillna(dataset.mean(), inplace=True)
	dataset = dataset.values

	splitting_data(dataset, split, trainingSet, testSet)
	print ('Train set: ' + repr(len(trainingSet)))
	print ('Test set: ' + repr(len(testSet)))
	k = int(input('Masukkan nilai K : '))
	for x in range(len(testSet)):
		neighbors = get_neighbors(trainingSet, testSet[x], k)
		result = get_response(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = get_accuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')