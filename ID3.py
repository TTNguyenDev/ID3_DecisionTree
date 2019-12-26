#!/usr/bin/python3
import os
import copy
import csv
import pandas as pd
from math import log2, fsum
from sklearn import metrics
INF = 99999

class Node:
	def __init__(self, parent, pk, nk, nodeType, attributeName=None, attributeIndex=-1, classification=None):
		self.parent = parent
		self.p = pk
		self.n = nk
		self.type = nodeType
		if(self.type == 'test'):
			self.attributeName = attributeName
			self.attributeIndex = attributeIndex
		else:
			self.classification = classification
		self.branches = {}


class DecisionTree:
	def __init__(self, attributes, examples):
		self.attributes = attributes
		self.examples = examples

		values = {}
		for index, a in enumerate(attributes):
			temp = []
			for row in examples:
				if(row[index] not in temp):
					temp.append(row[index])
			values[a] = temp

		self.attributeValues = values
		self.classLabel = self.attributeValues[self.attributes[len(attributes) - 1]] #class label
		self.pValue, self.nValue = self.classLabel[0], self.classLabel[1]
		self.p, self.n = self.count(examples)
		self.takenAttributes = []

	def ID3(self, examples, attributes, parent, parentExamples):
		if(len(examples) == 0):
			return self.pluralityValue(parent, parentExamples)
		elif(self.same_class(examples)):
			p, n = self.count(examples)
			return Node(parent, p, n, 'leaf', classification = self.pValue if p > 0 else self.nValue)
		elif((len(attributes) - len(self.takenAttributes)) == 0):
			return self.pluralityValue(parent, examples)
		else:
			attrIndex = self.maxIG(attributes, examples)
			attribute = attributes[attrIndex]
			p, n = self.count(examples)
			
			root = Node(parent, p, n, 'test', attributeName = attribute, attributeIndex = attrIndex)
			self.takenAttributes.append(attribute)

			for value in self.attributeValues[attribute]:
				newExample = []
				for row in examples:
					if(row[attrIndex] == value):
						newExample.append(copy.deepcopy(row))
				childNode = self.ID3(newExample, attributes, root, examples)

				root.branches[value] = childNode

		return root

	def same_class(self, examples,):
		prevValue = examples[0][len(examples[0]) - 1]
		for row in examples[1:]:
			if(row[len(row) - 1] != prevValue):
				return False
		return True
  
	def entropy_s(self, q):
		return q if q == 0.0 else -1 * (q * log2(q) + (((1 - q) * log2(1 - q)) if q < 1 else 0))
	
	def cal_entropy(self, counts):
		return ((counts[2] + counts[3])/(counts[0] + counts[1])) * self.entropy_s((counts[2]/(counts[2] + counts[3])) if (counts[2] + counts[3] > 0) else 0)

	def remainder(self,subsets):
		return fsum(map(self.cal_entropy, subsets))

	def maxIG(self, attributes, examples):
		maxVal = -INF
		maxValInd = -1
		
		for index, a in enumerate(attributes[:len(attributes) - 1]):
			if(a not in self.takenAttributes):
				gain = self.IG(a, index, examples)
				if(gain > maxVal):
					maxVal = gain
					maxValInd = index
		return maxValInd


	def IG(self, attribute, index, examples):
		subsets = []

		for value in self.attributeValues[attribute]:
			pk = nk = 0
			for row in examples:
				if(row[index] == value):
					if(row[len(row) - 1] == self.pValue):
						pk += 1
					else:
						nk += 1

			subsets.append((self.p, self.n, pk, nk))
		
		return self.entropy_s(self.p/(self.p + self.n)) - self.remainder(subsets)		

	def pluralityValue(self, parent, examples):
		p, n = self.count(examples)
	
		return Node(parent, p, n, 'leaf', classification = self.pValue if p > n else self.nValue)

	def count(self, examples):
		p = n = 0

		for row in examples:
			if(row[len(row) - 1] == self.pValue):
				p += 1
			elif(row[len(row) - 1] == self.nValue):
				n += 1

		return p, n

	def printDTree(self, node, value=None):
		print(node.parent.attributeName + '=' if node.parent else '', value if value else '')
		for branch in node.branches:
			if(node.branches[branch].type == 'leaf'):
				print('|' + node.branches[branch].parent.attributeName, ' = ', branch if branch else '', ':', node.branches[branch].classification)

		for branch in node.branches:
			if(node.branches[branch].type == 'test'):
				self.printDTree(node.branches[branch], branch)


#Cross validation: KFolds

#predict data
	def traverseTree(self, test, node):
		attributeValue = test[node.attributeName]
		if(node.branches[attributeValue].type == 'leaf'):
			return node.branches[attributeValue].classification
		else:
			return self.traverseTree(test, node.branches[attributeValue])

	def predict(self, testSet):
		predictions = []
		for index, row in enumerate(testSet):
			test = {}
			for index, data in enumerate(row):
				test[self.attributes[index]] = data

			predictions.append(self.traverseTree(test, self.root))

		return predictions


df = pd.read_csv('weather.csv')
columnNames = list(df.columns)
trainData = df.values
decisionTree = DecisionTree(columnNames, trainData)

decisionTree.root = decisionTree.ID3(trainData, columnNames, None, trainData)

print('===Classifier model(full training set)===')
decisionTree.printDTree(decisionTree.root)
newdf = df[df.columns[0:len(df.columns)-1]]
predictions = decisionTree.predict(newdf.values)

#Cross validation:
def crossValidation(df, n_split):
	size = len(df)/n_split
	predicted_arr = []
	expected_arr = []
	for i in range(0, len(df), int(size)):
		start = i
		end = i + int(size)
	
		y = df[start:end]
		X = df.drop(y.index)

		decisionTree = DecisionTree(columnNames, X.values)
		decisionTree.root = decisionTree.ID3(X.values, columnNames, None, X.values)
		X_test = y[y.columns[0:len(df.columns)-1]]
		y_test = y.iloc[:,len(df.columns)-1].values
		
		predictions = decisionTree.predict(X_test.values)

		predicted_arr.extend(predictions)
		expected_arr.extend(y_test)

	return predicted_arr, expected_arr

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct, correct / float(len(actual)) * 100.0

predicted, y_test = crossValidation(df, 3)
correct_label, percent = accuracy_metric(y_test, predicted)

print("Correctly Classified Instances", correct_label, str(percent) + '%')
print("Incorrectly Classified Instances", str(len(y_test)- correct_label), str(100-percent) + '%')

print("===Detailed Accuracy By Class===")

confusion = metrics.confusion_matrix(y_test, predicted)
TN, FP    = confusion[0, 0], confusion[0, 1]
FN, TP    = confusion[1, 0], confusion[1, 1]
tp_precision = TP/(TP+FP)
fp_precision = TN/(TN+FN)
tp_recall = TP/(TP+FN)
fp_recall = TN/(TN+FP)
tp_fmeasure = (2*tp_recall*tp_precision)/(tp_recall+tp_precision)
fp_fmeasure = (2*fp_recall*fp_precision)/(fp_recall+fp_precision)
print('Class', 'TP Rate', 'FP Rate', 'Precision', 'Recall', 'F-Measure')
print('yes', TP, FP, tp_precision, tp_recall, tp_fmeasure)
print('no', TN, FN, fp_precision, fp_recall, fp_fmeasure)
