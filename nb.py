#!/usr/bin/env python

import sys,math,copy,random

class Data:
	def __init__(self):
		#label really means class
		self.label = 0
		self.attributes = []
		self.attrvalues = []

	def getLabelSign(self):
		if self.label == 0:
			return -1
		return 1

	def setLabel(self,l):
		self.label = l

	def setAttribute(self,attr,value):
		self.attributes.append(attr)
		self.attrvalues.append(value)

class NaiveBayes:
	def __init__(self,training,test,beta,model):
		self.base = 0
		self.training = self.read_csv(training)
		self.beta = float(beta)
		self.weights = self.computeMAP();
		self.export_model(model)
		accuracy = self.test(test)
		#print("Training Accuracy is " + str(self.test(training)))
		print("Test Accuracy is " + str(accuracy))

	def computeMAP(self):
		weights = []

		#calculate the weights for each attribute
		for i in range(len(self.training[0].attributes)):
			wi = math.log(self.prob(i,1,1)/self.prob(i,1,0)) - math.log(self.prob(i,0,1)/self.prob(i,0,0))
			weights.append(wi)

		#calculate the base weight
		self.base = math.log(self.proby(1)/self.proby(0))
		for i in range(len(weights)):
			self.base += math.log(self.prob(i,0,1)/self.prob(i,0,0))

		return weights

	def proby(self,y): #P(Y = y)
		yi = 0 ## for Y = y
		yt = 0 ## for Y total
		for i in range(len(self.training)):
			if self.training[i].label == y:
				yi+=1
			yt+=1
		return (yi+self.beta-1)/(yt+(2*self.beta)-2)

	def prob(self,i,x,y): #P(Xi = x | Y = y)

		xx = 0 #number of items for which Xi = x and Y = y
		yy = 0 #number of items for wihch Y = y

		for j in range(len(self.training)): #loops every example
			if self.training[j].label == y:
				yy+=1
				if self.training[j].attrvalues[i] == x:
					xx+=1

		return (xx+self.beta-1)/(yy+(2*self.beta)-2)


	def predict_answer(self,i,test):
		
		probsum = self.base
		for i in range(len(self.weights)):
			if test.attrvalues[i] == 1:
				probsum += self.weights[i]
		prob = 1/(1+math.exp(-probsum))
		print(" "+str(prob))
		if prob >= 0.5:
			return True
		return False

	def read_csv(self,fn):
		data = []
		attrs = []
		f = open(fn,'r')
		firstline = True
		for line in f:
			line = line.strip()
			l = line.split(',')
			if firstline == True:
				firstline = False
				for attr in l:
					if (attr == 'spam'):
						continue
					attrs.append(attr)
				continue
			dataobj = Data()
			for i in range(len(attrs)):
				dataobj.setAttribute(attrs[i],int(l[i]))
			dataobj.setLabel(int(l[i+1]))
			data.append(dataobj)
		f.close()
		if len(data) == 0:
			print("NO DATA")
			sys.exit()
		return data

	def export_model(self,fn):
		f = open(fn,'w')
		f.write(str(str(self.base)+'\n'))
		for i in range(len(self.weights)):
			f.write(str(self.training[0].attributes[i])+" "+str(self.weights[i])+"\n")
		f.close()
		pass

	def test(self,filename):
		correct = 0
		total = 0
		testdata = self.read_csv(filename)
		for i in range(len(testdata)):
			if self.predict_answer(i,testdata[i]) == testdata[i].label:
				correct+= 1
			total += 1
		return float(correct)/float(total)

def main():
	#./nb <train> <test> <beta> <model>
	NaiveBayes(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

main()
