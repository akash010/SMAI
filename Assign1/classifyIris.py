import csv
import sys
import random
import math
import numpy

def randomSubSampling(dataset):
	Testdataset = []
	Traindataset = []
	random.shuffle(dataset)
	for i in range(len(dataset)):
		if i%2 ==0:
			Testdataset.append(dataset[i])
		else:
			Traindataset.append(dataset[i])
	return [Testdataset, Traindataset]

def classifyInstance1NN(Traindataset, inst):
	distVect = {}
	dist = 0
	for row in Traindataset:
		for i in range(len(row)):
			try:
				dist = dist + (row[i]-inst[i])*(row[i]-inst[i])
			except:
				className = row[i]
		dist = math.sqrt(dist)
		distVect[dist] = className
		dist = 0		
	key = sorted(distVect)[0]
	return distVect[key]

def classifyInstance3NN(Traindataset, inst):
	distVect = {}
	dist = 0
	for row in Traindataset:
		for i in range(len(row)):
			try:
				dist = dist + (row[i]-inst[i])*(row[i]-inst[i])
			except:
				className = row[i]
		dist = math.sqrt(dist)
		distVect[dist] = className
		dist = 0		
	key = sorted(distVect)[0]
	return distVect[key]

def applyKNNSubSampling(dataset, classes, classColumnNum, K):
	Testdataset, Traindataset = randomSubSampling(dataset)

	confusionMatrix = [[key,[0,0,0]] for key in classes.keys()]
	confusionMatrix = dict(confusionMatrix)
	
	for row in Testdataset:
		if k==1:
			clas = classifyInstance1NN(Traindataset, row)
		else:
			clas = classifyInstance3NN(Traindataset, row)

		confusionMatrix[row[classColumnNum]][classes[clas]] = confusionMatrix[row[classColumnNum]][classes[clas]]+1
	
	return confusionMatrix

def extractClasses(dataset, classColumnNum):
	col = [row[classColumnNum] for row in dataset]
	col = list(set(col))
	col = [[col[i],i] for i in range(len(col))]
	return col

def getInputArgs():
	try:
		fileName = sys.argv[1]
		classColumnNum = int(sys.argv[2])
	except:
		print "arguments : <fileName> <classColumnNumber>"
		sys.exit()

	return [fileName, classColumnNum]

def getDataset(fileName):
	dataset = []
	with open(fileName, 'rb') as f:
	    reader = csv.reader(f)

	    for row in reader:
	    	tmp = []
	    	if len(row) == 0:
	    		continue
	    	for i in range(len(row)):
			try:
				tmp.append(float(row[i]))
			except:
				tmp.append(row[i])
		dataset.append(tmp)
	f.close()
	return dataset

def getAccuracy(confusionMatrix,classes):
	correct=0
	incorrect=0
	for key in confusionMatrix.keys():
		for i in range(len(classes.keys())):
			if classes[key] == i:
				correct = correct + confusionMatrix[key][i]
			else:
				incorrect = incorrect + confusionMatrix[key][i]		
	total = correct+incorrect
	return (correct/float(total))*100

def calculate1NN(dataset, classes, classColumnNum):
	FinConfusionMat = []
	dimension = 0
	stats = []

	for k in range(10):
		confusionMatrix = applyKNNSubSampling(dataset, classes, classColumnNum,1)
		stats.append(getAccuracy(confusionMatrix,classes))
		if k == 0:
			FinConfusionMat = confusionMatrix
		else:
			for key in confusionMatrix.keys():
				for i in range(len(classes.keys())):
					FinConfusionMat[key][i] = FinConfusionMat[key][i]+confusionMatrix[key][i]

	arr = numpy.array(stats)
	print "\n**************************"
	print "Final confusion Matrix for 1NN with 10 iterations with randomSubSampling:\n"
	print FinConfusionMat

	print ""
	print "Mean: " + str(arr.mean())
	print "Standard Deviation: " + str(arr.std())
	print "**************************\n"

def calculate3NN(dataset, classes, classColumnNum):
	FinConfusionMat = []
	dimension = 0
	stats = []

	for k in range(10):
		confusionMatrix = applyKNNSubSampling(dataset, classes, classColumnNum,3)
		stats.append(getAccuracy(confusionMatrix,classes))
		if k == 0:
			FinConfusionMat = confusionMatrix
		else:
			for key in confusionMatrix.keys():
				for i in range(len(classes.keys())):
					FinConfusionMat[key][i] = FinConfusionMat[key][i]+confusionMatrix[key][i]

	arr = numpy.array(stats)
	print "\n**************************"
	print "Final confusion Matrix for 3NN with 10 iterations with randomSubSampling:\n"
	print FinConfusionMat

	print ""
	print "Mean: " + str(arr.mean())
	print "Standard Deviation: " + str(arr.std())
	print "**************************\n"


def main():
	fileName, classColumnNum = getInputArgs()
	dataset = getDataset(fileName)
	classes = extractClasses(dataset, classColumnNum)
	classes = dict(classes)

	calculate1NN(dataset, classes, classColumnNum)
	calculate3NN(dataset, classes, classColumnNum)

	del dataset


main()


