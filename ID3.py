#!/usr/bin/python3
import os
import copy
import csv
import pandas as pd
from math import log2, fsum

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
		self.branches = {} #key of branches will be the different values for the attribute


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
		#detect same class
		same_class = True
		prevValue = examples[0][len(examples[0]) - 1]
		for row in examples[1:]:
			if(row[len(row) - 1] != prevValue):
				same_class = False


		if(len(examples) == 0):
			return self.pluralityValue(parent, parentExamples)
		elif(same_class):
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
		'''
		Returns a leaf node with majority class value
		'''
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
			if (index == 0):
				continue
			test = {}

			for index, data in enumerate(row):
				test[self.attributes[index]] = data

			predictions.append(self.traverseTree(test, self.root))

		return predictions

with open('weather.csv') as csvFile:
	dataset = csv.reader(csvFile, delimiter=',')

	firstline = 1
	train = []
	for row in dataset:
		if(firstline):
			firstline = 0
			columnNames = copy.deepcopy(row)
		else:
			train.append(copy.deepcopy(row))

	decisionTree = DecisionTree(columnNames, train)

	decisionTree.root = decisionTree.ID3(train, columnNames, None, train)
	
	print('===Classifier model(full training set)===')
	decisionTree.printDTree(decisionTree.root)
