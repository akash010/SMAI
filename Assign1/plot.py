import csv
import sys
import matplotlib.pyplot as plt

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

fileName, classColumnNum = getInputArgs()
dataset = getDataset(fileName,classColumnNum)

x = []
y = []

for i in range(len(dataset)):
	for j in range(len(dataset[i])):
		if j ==1:
			x.append(dataset[i][j])
		if j == 3:
			y.append(dataset[i][j])

plt.plot(x, y, 'ro')

plt.show()