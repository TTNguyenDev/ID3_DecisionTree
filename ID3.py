#!/usr/bin/python3
import copy
import pandas as pd
from numpy import log2
import numpy as np 
from sklearn import metrics
import argparse
INF = 99999
class Node:
	def __init__(self, parent, posLabel, negLabel, type, attributeName=None, attributeIndex=-1, classification=None):
		self.parent = parent
		self.posLabel = posLabel
		self.negLabel = negLabel
		self.type = type
		if(self.type == 'nonLeaf'):
			self.attributeName = attributeName
			self.attributeIndex = attributeIndex
		else:
			self.classification = classification
		self.branches = {}


class DecisionTree:
	def __init__(self, attributes, df):
		self.attributes = attributes
		self.df = df.values

		values = {}
		for index, a in enumerate(attributes):
			values[a] = df[a].unique().tolist()
		self.attributeValues = values
		self.classLabel = self.attributeValues[self.attributes[len(attributes) - 1]] #class label
		self.pValue, self.nValue = self.classLabel[0], self.classLabel[1]
		self.p, self.n = self.countClassLabel(df.values)
		self.takenAttributes = []

	def ID3(self, df, attributes, parent, parentExamples):
		if(len(df) == 0):
			return self.predictNode(parent, parentExamples)
		elif(self.same_class(df) or (len(attributes) - len(self.takenAttributes)) == 0):
			return self.predictNode(parent, df)
		else:
			attrIndex = self.find_winner(attributes, df)
			attribute = attributes[attrIndex]
			p, n = self.countClassLabel(df)
			
			root = Node(parent, p, n, 'nonLeaf', attributeName = attribute, attributeIndex = attrIndex)
			self.takenAttributes.append(attribute)

			for value in self.attributeValues[attribute]:
				newExample = []
				for row in df:
					if(row[attrIndex] == value):
						newExample.append(copy.deepcopy(row))
				childNode = self.ID3(newExample, attributes, root, df)
				root.branches[value] = childNode

		return root

	def same_class(self, df):
		prevValue = df[0][len(df[0]) - 1]
		for row in df[1:]:
			if(row[len(row) - 1] != prevValue):
				return False
		return True
  
	def find_entropy(self, q):
		if q == 0:
			return 0
		if q >= 1:
			return -1 * (q * log2(q))
		temp = 1-q
		return -1 * (q * log2(q) + temp*log2(temp))

	def find_winner(self, attributes, examples):
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
		i_attr = 0
		for value in self.attributeValues[attribute]:
			pk = nk = 0
			for row in examples:
				if(row[index] == value):
					if(row[len(row) - 1] == self.pValue):
						pk += 1
					else:
						nk += 1
			
			i_attr += ((pk + nk)/(self.p + self.n)) * self.find_entropy((pk/(pk + nk)) if (pk + nk > 0) else 0)
		h = self.find_entropy(self.p/(self.p + self.n))
		return h - i_attr	

	def predictNode(self, parent, df):
		p, n = self.countClassLabel(df)
		return Node(parent, p, n, 'leaf', classification = self.pValue if p > n else self.nValue)

	def countClassLabel(self, examples):
		return sum(list(x).count(self.pValue) for x in examples), sum(list(x).count(self.nValue) for x in examples)

	def printDTree(self, node, value=None):
		print(node.parent.attributeName + '=' if node.parent else '', value if value else '')
		
		for branch in node.branches:
			if(node.branches[branch].type == 'leaf'):
				print('|' + node.branches[branch].parent.attributeName, ' = ', branch if branch else '', ':', node.branches[branch].classification)
				
		for branch in node.branches:
			if(node.branches[branch].type == 'nonLeaf'):
				self.printDTree(node.branches[branch], branch)

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

#Cross validation:
def crossValidation(df, n_split):
	size = len(df)/n_split
	predicted_arr = []
	expected_arr = []

	for i in range(0, len(df), int(size)):
		start = i
		end = i + int(size)
		print(start, end)
		y = df[start:end]
		X = df.drop(y.index)

		print(y)
		print(X)
		columnNames = df.columns
		decisionTree = DecisionTree(columnNames, X)
		decisionTree.root = decisionTree.ID3(X.values, columnNames, None, X.values)
		X_test = y[y.columns[0:len(df.columns)-1]]
		y_test = y.iloc[:,len(df.columns)-1].values
		
		predictions = decisionTree.predict(X_test.values)

		predicted_arr.extend(predictions)
		expected_arr.extend(y_test)

	return predicted_arr, expected_arr

def drawID3(node, value, wf):
		print(node.parent.attributeName + '=' if node.parent else '', value if value else '')
		if (node.parent):
			wf.write(node.parent.attributeName + '=')
		if (value):
			wf.write(value+'\n')
		for index, branch in enumerate(node.branches):
			if(node.branches[branch].type == 'leaf'):
				print('|' + node.branches[branch].parent.attributeName, ' = ', branch if branch else '', ':', node.branches[branch].classification)
				wf.write('|' + node.branches[branch].parent.attributeName + ' = ')
				if (branch):
					wf.write(branch)
				wf.write(':' + node.branches[branch].classification + '\n')
		for branch in node.branches:
			if(node.branches[branch].type == 'nonLeaf'):
				drawID3(node.branches[branch], branch, wf)

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct, correct / float(len(actual)) * 100.0

# main function
parser=argparse.ArgumentParser()
parser.add_argument('--input', help='Name of csv file')
parser.add_argument('--output', help='output')
parser.add_argument('--folds', help='folds')

args=parser.parse_args()
input = args.input
output = args.output
folds = int(args.folds)

df = pd.read_csv(input)
wf = open(output,"w")
columnNames = list(df.columns)
trainData = df.values

decisionTree = DecisionTree(columnNames, df)
decisionTree.root = decisionTree.ID3(trainData, columnNames, None, trainData)

print('===Classifier model(full training set)===')
wf.write('===Classifier model(full training set)===\n')
drawID3(decisionTree.root, None, wf)

print(folds)
predicted, y_test = crossValidation(df, folds)
correct_label, percent = accuracy_metric(y_test, predicted)

print("Correctly Classified Instances", correct_label, '{:10.2f}'.format(percent) + '%')
print("Incorrectly Classified Instances", str(len(y_test)- correct_label), '{:10.2f}'.format(100-percent) + '%')
wf.write("Correctly Classified Instances\t" + str(correct_label) + '\t' + '{:10.2f}'.format(percent) + '%\n')
wf.write("Incorrectly Classified Instances\t"+ str(len(y_test)- correct_label) + '\t' + '{:10.2f}'.format(100-percent) + '%\n')

print("===Detailed Accuracy By Class===")
wf.write("===Detailed Accuracy By Class===\n")

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
wf.write('Class\t'+ 'TP Rate\t\t'+ 'FP Rate\t\t'+ 'Precision\t\t'+ 'Recall\t\t'+ 'F-Measure\n')
print('yes', TP, FP, tp_precision, tp_recall, tp_fmeasure)
print('no', TN, FN, fp_precision, fp_recall, fp_fmeasure)
wf.write('yes\t' + '{:10.2f}'.format(TP) +'\t'+ '{:10.2f}'.format(FP)+'\t'+ '{:10.2f}'.format(tp_precision) +'\t' + '{:10.2f}'.format(tp_recall)+'\t'+ '{:10.2f}'.format(tp_fmeasure)+'\n')
wf.write('no\t' + '{:10.2f}'.format(TN) +'\t'+ '{:10.2f}'.format(FN)+'\t'+ '{:10.2f}'.format(fp_precision) +'\t' + '{:10.2f}'.format(fp_recall)+'\t'+ '{:10.2f}'.format(fp_fmeasure)+'\n')
wf.close()

