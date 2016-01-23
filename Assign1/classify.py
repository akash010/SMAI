import csv
import sys
import random
import math
import numpy
import matplotlib.pyplot as plt
import operator

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

def fiveFoldVerif(dataset,kthfold):
	Testdataset = []
	Traindataset = []
	random.shuffle(dataset)
	dim = len(dataset)

	foldsize = dim/5

	for i in range(len(dataset)):
		if i > foldsize*kthfold and i < foldsize*(kthfold+1):
			Testdataset.append(dataset[i])
		else:
			Traindataset.append(dataset[i])
	return [Testdataset, Traindataset]

def classifyInstanceNN(Traindataset, inst, classes,k):
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
	keys = sorted(distVect)
	counts={}

	for i in range(k):
		counts[distVect[keys[i]]] = 0		

	for i in range(k):
		counts[distVect[keys[i]]] = counts[distVect[keys[i]]] +1

	mx = -1
	clas = None

	for key in counts.keys():
		if counts[key] > mx :
			mx = counts[key]
			clas = key
	return clas

def applyKNN(dataset, classes, classColumnNum, k, flag, kthfold):
	if flag:
		Testdataset, Traindataset = randomSubSampling(dataset)
	else:
		Testdataset, Traindataset = fiveFoldVerif(dataset, kthfold)

	confusionMatrix = [[key,[0,0,0]] for key in classes.keys()]
	confusionMatrix = dict(confusionMatrix)
	
	for row in Testdataset:
		clas = classifyInstanceNN(Traindataset, row, classes,k)
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

def getDataset(fileName,classColumnNum):
	dataset = []
	with open(fileName, 'rb') as f:
	    reader = csv.reader(f)

	    for row in reader:
	    	tmp = []
	    	if len(row) == 0:
	    		continue
	    	for i in range(len(row)):
			try:
				if i == classColumnNum:
					tmp.append(row[i])
				else:
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

def calculate1NNRand(dataset, classes, classColumnNum):
	FinConfusionMat = []
	dimension = 0
	stats = []

	for k in range(10):
		confusionMatrix = applyKNN(dataset, classes, classColumnNum,1, True,0)
		stats.append(getAccuracy(confusionMatrix,classes))
		if k == 0:
			FinConfusionMat = confusionMatrix
		else:
			for key in confusionMatrix.keys():
				for i in range(len(classes.keys())):
					FinConfusionMat[key][i] = FinConfusionMat[key][i]+confusionMatrix[key][i]

	arr = numpy.array(stats)
	print "\n**************************"
	print "Final confusion Matrix for 1NN with 10 iterations of randomSubSampling:\n"
	print FinConfusionMat

	print ""
	print "Mean: " + str(arr.mean())
	print "Standard Deviation: " + str(arr.std())
	print "**************************\n"

def calculate3NNRand(dataset, classes, classColumnNum):
	FinConfusionMat = []
	dimension = 0
	stats = []

	for k in range(10):
		confusionMatrix = applyKNN(dataset, classes, classColumnNum,3, True,0)
		stats.append(getAccuracy(confusionMatrix,classes))
		if k == 0:
			FinConfusionMat = confusionMatrix
		else:
			for key in confusionMatrix.keys():
				for i in range(len(classes.keys())):
					FinConfusionMat[key][i] = FinConfusionMat[key][i]+confusionMatrix[key][i]

	arr = numpy.array(stats)
	print "\n**************************"
	print "Final confusion Matrix for 3NN with 10 iterations of randomSubSampling:\n"
	print FinConfusionMat

	print ""
	print "Mean: " + str(arr.mean())
	print "Standard Deviation: " + str(arr.std())
	print "**************************\n"

def calculate1NNfold(dataset, classes, classColumnNum):
	FinConfusionMat = []
	dimension = 0
	stats = []

	for k in range(5):
		confusionMatrix = applyKNN(dataset, classes, classColumnNum,1, False, k)
		stats.append(getAccuracy(confusionMatrix,classes))
		if k == 0:
			FinConfusionMat = confusionMatrix
		else:
			for key in confusionMatrix.keys():
				for i in range(len(classes.keys())):
					FinConfusionMat[key][i] = FinConfusionMat[key][i]+confusionMatrix[key][i]

	arr = numpy.array(stats)
	print "\n**************************"
	print "Final confusion Matrix for 1NN with 5 fold cross validation:\n"
	print FinConfusionMat

	print ""
	print "Mean: " + str(arr.mean())
	print "Standard Deviation: " + str(arr.std())
	print "**************************\n"

def calculate3NNfold(dataset, classes, classColumnNum):
	FinConfusionMat = []
	dimension = 0
	stats = []

	for k in range(5):
		confusionMatrix = applyKNN(dataset, classes, classColumnNum,3, False, k)
		stats.append(getAccuracy(confusionMatrix,classes))
		if k == 0:
			FinConfusionMat = confusionMatrix
		else:
			for key in confusionMatrix.keys():
				for i in range(len(classes.keys())):
					FinConfusionMat[key][i] = FinConfusionMat[key][i]+confusionMatrix[key][i]

	arr = numpy.array(stats)
	print "\n**************************"
	print "Final confusion Matrix for 3NN with 5 fold cross validation:\n"
	print FinConfusionMat

	print ""
	print "Mean: " + str(arr.mean())
	print "Standard Deviation: " + str(arr.std())
	print "**************************\n"

def decisonLine(dataPoints, plt):	

	ln1=[]
	ln2=[]
	ln3=[]
	ln4=[]

	del dataPoints[-1]
	dataPoints = sorted(dataPoints, key = operator.itemgetter(3))
	dataPoints = sorted(dataPoints, key = operator.itemgetter(1))

	point1={}
	point2={}

	ind=0
	length=len(dataPoints)

	for var in range(45):
		x=float(var)/10

		if ind >= length:
			break

		temp=""

		while dataPoints[ind][1]==str(x):
			if temp!="" and temp!=dataPoints[ind][4]:
				if temp=='Iris-setosa':
					point1[x]=float(dataPoints[ind-1][3])
				elif temp=='Iris-versicolor':
					point2[x]=float(dataPoints[ind-1][3])

			temp=dataPoints[ind][4]			
			ind+=1
			if ind >= length:
				break

		if temp=='Iris-setosa':
			point1[x]=float(dataPoints[ind-1][3])
		elif temp=='Iris-versicolor':
			point2[x]=float(dataPoints[ind-1][3])

	point1[0]=0.25
	point2[0]=1
	point1[4.5]=0.5
	point2[4.5]=1.5

	for var in range(46):
		x=float(var)/10
		if x in point1:
			ln1.append(x)
			ln2.append(point1[x])

		if x in point2:
			ln3.append(x)
			ln4.append(point2[x])	

	plt.plot(ln1,ln2,'r')
	plt.plot(ln3,ln4,'g')
	plt.legend(loc='upper left')
	plt.axis([0, 4.3, 0, 2.8])
	plt.xlabel("widthSepal")
	plt.ylabel("widthPetal")
	plt.show()	

def plotGraph(dataPoints):
	widthSep = []
	temp = []
	for i in range(3):
		for j in range(50):
			temp.append(0)
		widthSep.append(temp)
		temp = []

	widthPet = []
	temp = []
	for i in range(3):
		for j in range(50):
			temp.append(0)
		widthPet.append(temp)
		temp = []

	listClasses=['Iris-setosa','Iris-versicolor','Iris-virginica']

	for i in range(3):
		j=0
		for k in range(len(dataPoints) - 1):
			temp=dataPoints[k]
			if temp[4]==listClasses[i]:
				widthPet[i][j]=temp[3]
				widthSep[i][j]=temp[1]
				j= j + 1

	plt.plot(widthSep[0],widthPet[0], 'ro',label='Iris-setosa')
	plt.plot(widthSep[1],widthPet[1], 'go', label='Iris-versicolor')
	plt.plot(widthSep[2],widthPet[2], 'bo',label='Iris-virginica')

	decisonLine(dataPoints, plt)

def main():
	fileName, classColumnNum = getInputArgs()
	dataset = getDataset(fileName,classColumnNum)
	classes = extractClasses(dataset, classColumnNum)
	classes = dict(classes)

	calculate1NNRand(dataset, classes, classColumnNum)
	calculate3NNRand(dataset, classes, classColumnNum)

	calculate1NNfold(dataset, classes, classColumnNum)
	calculate3NNfold(dataset, classes, classColumnNum)

	del dataset

	with open(fileName, 'rb') as f:
		reader = csv.reader(f)
		dataset = list(reader)
		plotGraph(dataset)	

	del dataset

main()
