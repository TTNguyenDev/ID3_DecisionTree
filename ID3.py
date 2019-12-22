#!/usr/bin/python3
import os
import copy
import csv
from math import log2, fsum, inf

class Utilities:
	B = lambda q : q if q == 0.0 else -1 * (q * log2(q) + (((1 - q) * log2(1 - q)) if q < 1 else 0))
	cal_entropy = lambda counts : ((counts[2] + counts[3])/(counts[0] + counts[1])) * Utilities.B((counts[2]/(counts[2] + counts[3])) if (counts[2] + counts[3] > 0) else 0)
	Sum = lambda list : fsum(list)
	Remainder = lambda subsets : Utilities.Sum(map(Utilities.cal_entropy, subsets))
	cal_ig = lambda p, n, subsets : Utilities.B(p/(p + n)) - Utilities.Remainder(subsets)

class DTNode:
	def __init__(self, parent, pk, nk, nodeType, attributeName=None, attributeIndex=-1, classification=None):
		self.parent = parent
		self.p = pk
		self.n = nk
		self.type = nodeType
		if(self.type == 'testNode'):
			self.attributeName = attributeName
			self.attributeIndex = attributeIndex
		else:
			self.classification = classification
		self.branches = {} #key of branches will be the different values for the attribute


class DecisionTree:
	def __init__(self, dataset):
		i = 0
		self.examples = []
		for row in dataset:
			if(i == 0):
				i = 1
				self.attributes = copy.deepcopy(row)
			else:
				self.examples.append(copy.deepcopy(row))
		
		self.attributeValues = self.getAttributeValues(self.attributes, self.examples)
		targetVals = self.attributeValues[self.attributes[len(self.attributes) - 1]]
		self.pValue, self.nValue = targetVals[0], targetVals[1]
		self.p, self.n = self.getClassCount(self.examples)
		self.takenAttributes = []


	def ID3(self, examples, attributes, parent, parentExamples):
		'''
		Implementation of the DTL algorithm
		'''
		if(len(examples) == 0):
			#return a leaf node with the majority class value in parent examples
			return self.pluralityValue(parent, parentExamples)
		elif(self.hasSameClass(examples)):
			#return a leaf node with the class value
			p, n = self.getClassCount(examples)
			return DTNode(parent, p, n, 'leafNode', classification = self.pValue if p > 0 else self.nValue)
		elif((len(attributes) - len(self.takenAttributes)) == 0):
			#return a leaf node with the majority class value in examples
			return self.pluralityValue(parent, examples)
		else:
			#find the attribute that has max information gain
			attrIndex = self.maxIG(attributes, examples)
			attribute = attributes[attrIndex]
			p, n = self.getClassCount(examples)

			#create a root node
			root = DTNode(parent, p, n, 'testNode', attributeName = attribute, attributeIndex = attrIndex)
			#to track the attributes in inner nodes
			self.takenAttributes.append(attribute)

			#divide the examples and recursively call DTL to create child nodes
			for value in self.attributeValues[attribute]:
				newExample = []
				for row in examples:
					if(row[attrIndex] == value):
						newExample.append(copy.deepcopy(row))
				childNode = self.ID3(newExample, attributes, root, examples)

				#add the sub tree to the main tree
				root.branches[value] = childNode

		return root



	def maxIG(self, attributes, examples):
		'''
		Calculate the Importance value or the information gain for all attributes
		Return the attribute with max gain
		'''
		maxVal = -inf
		maxValInd = -1
		
		for index, a in enumerate(attributes[:len(attributes) - 1]):
			if(a not in self.takenAttributes):
				gain = self.IG(a, index, examples)
				if(gain > maxVal):
					maxVal = gain
					maxValInd = index

		return maxValInd


	def IG(self, attribute, index, examples):
		'''
		Calculate the gain for a given attribute
		'''
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

		return Utilities.cal_ig(self.p, self.n, subsets)


	def getAttributeValues(self, attributes, examples):
		'''
		To find the domain values for each attribute
		'''
		values = {}

		for index, a in enumerate(attributes):
			temp = []
			for row in examples:
				if(row[index] not in temp):
					temp.append(row[index])
			values[a] = temp

		return values


	def hasSameClass(self, examples):
		prevValue = examples[0][len(examples[0]) - 1]
		
		for row in examples[1:]:
			if(row[len(row) - 1] != prevValue):
				return False

		return True

	def pluralityValue(self, parent, examples):
		'''
		Returns a leaf node with majority class value
		'''
		p, n = self.getClassCount(examples)
		return DTNode(parent, p, n, 'leafNode', classification = self.pValue if p > n else self.nValue)

	def getClassCount(self, examples):
		'''
		Returns the number of examples in positive and negative classes
		'''
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
			if(node.branches[branch].type == 'leafNode'):
				print('|' + node.branches[branch].parent.attributeName, ' = ', branch if branch else '', ':', node.branches[branch].classification)

		for branch in node.branches:
			if(node.branches[branch].type == 'testNode'):
				self.printDTree(node.branches[branch], branch)

#predict data
	def traverseTree(self, test, node):
		attributeValue = test[node.attributeName]
		if(node.branches[attributeValue].type == 'leafNode'):
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


if (__name__ == '__main__'):

	decisionTree = None
	with open('weather.csv') as csvFile:
		dataset = csv.reader(csvFile, delimiter=',')
		decisionTree = DecisionTree(dataset)
		decisionTree.root = decisionTree.ID3(decisionTree.examples, decisionTree.attributes, None, decisionTree.examples)
		print('===Classifier model(full training set)===')
		decisionTree.printDTree(decisionTree.root)
