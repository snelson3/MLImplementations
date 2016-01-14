#!/usr/bin/env python
import sys,math,copy
MAXITER = 100
#implement perceptron algorithm

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

class Perceptron:
	def __init__(self,training,test,eta,model):
		self.eta = float(eta)
		self.data = self.read_csv(training)
		self.weights = self.optimize_perceptron()
		self.export_model(model)
		accuracy = self.test(test)
		#print("Training Accuracy is " + str(self.test(training)))
		print("Test Accuracy is " + str(accuracy))

	def update_weights(self,weights,output,example):
		#update weights in the case of a bad example
		for j in range(len(weights)-1):
			weights[j] += self.eta*(example.getLabelSign()-output)*example.attrvalues[j]
		weights[-1] += self.eta*(example.getLabelSign()-output)#update bias 

	def compute_activation(self,weights,example):
		activation = 0
		for j in range(len(weights)-1):
			activation += weights[j]*example.attrvalues[j]
		activation += weights[-1] #bias
		return activation

	def optimize_perceptron(self):
		weights = []
		for i in range(len(self.data[0].attributes)):
			weights.append(0)
		weights.append(0) #adding bias
	
		for iter in range(MAXITER):
			converged = True
			for example in self.data:
				activation = self.compute_activation(weights,example)
				if activation*example.getLabelSign() <= 0:
					converged = False
					self.update_weights(weights,activation,example)
			if converged == True:
				return weights
		return weights

	def predict_answer(self,test):
		activation = self.compute_activation(self.weights,test)
		return activation > 0

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
			f.write(str(self.data[0].attributes[i])+" "+str(self.weights[i])+"\n")
		f.close()
		pass

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
	#./perceptron <train> <test> <eta> <model>
	tree = Perceptron(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

main()
