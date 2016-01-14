#!/usr/bin/env python

import sys,math,copy

MAXITER = 100
EPSILON = 10.00001 #it won't converge/stop unless my epsilon is sufficiently big

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

class LogisticRegression:
	def __init__(self, training, test, eta, sigma, model):
		self.eta = float(eta)
		self.training = self.read_csv(training);
		self.sigmaconst = 1/(float(sigma)*float(sigma))
		self.weights = self.optimize()
		self.export_model(model)
		accuracy = self.test(test)
		#print("Training Accuracy is " + str(self.test(training)))
		print("Test Accuracy is " + str(accuracy))

	def optimize(self):
		print("optimizing")
		weights = []
		oldweights = []
		for i in range(len(self.training[0].attributes)):
			weights.append(0)
			oldweights.append(0)
		weights.append(0)
		oldweights.append(0)#accounting for the bias
		gradient = EPSILON+1
		itr = 0
		while (gradient > EPSILON) and (itr < MAXITER):
			oldweights = []
			for i in range(len(weights)):
				oldweights.append(weights[i])
			itr+=1
			for i in range(len(weights)):
				if i == len(weights)-1:
					weights[i] = oldweights[i] + self.eta*self.sum_prob(i,oldweights)
				else:
					weights[i] = (self.sigmaconst*oldweights[i]) + self.eta * self.sum_prob(i,oldweights)

			gradient = self.calcGradient(oldweights,weights)
			print("change " + str(itr) +" is " + str(gradient))
		return weights

	def calcGradient(self,oldwts,newwts):
		sm = 0
		for i in range(len(oldwts)-1):
			if (i == 0):
				continue
			wtchange = newwts[i]-oldwts[i]
			sm+= wtchange*wtchange
		return math.sqrt(sm)

	def pY(self,example,weights):
		sm = 0
		for i in range(len(example.attrvalues)): #its going to loop through a bunch of 0's ones
			sm += weights[i]*example.attrvalues[i]
		exponent = weights[-1] + sm
		if(exponent > 700):
			exponent = 700
		return math.exp(exponent) #if this is true it should be 0

	def pY1(self,example,weights):
		return True == (1 < self.pY(example,weights))

	def sum_prob(self,i,weights):
		num = 0
		for j in range(len(self.training)):
			if (i == len(weights)-1):
				jth = 1
			else:
				jth = self.training[j].attrvalues[i]
			jth = jth * (self.training[j].label - self.pY1(self.training[j],weights))
			num+=jth
		#print(num)
		return num

	def predict_answer(self,test):
		prob = self.pY(test,self.weights)
		print(prob)
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
		f.write(str(self.weights[-1])+'\n')
		for i in range(len(self.weights)-1):
			f.write(str(self.training[0].attributes[i])+" "+str(self.weights[i])+"\n")
		f.close()

	def test(self,filename):
		correct = 0
		total = 0
		testdata = self.read_csv(filename)
		for i in testdata:
			if self.predict_answer(i) == i.label:
				correct+= 1
			total += 1
		return float(correct)/float(total)

def main():
	#./logistic <train> <test> <eta> <sigma> <model>
	LogisticRegression(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

main()
