#!/usr/bin/env python
#NOTE TO GRADER
#NOTE TO GRADER
#I Did not get my chi_squared function working correctly (function should_stop() under class Tree)
# but I left it in so I could get partial credit for that part of the assignment
#NOTE TO GRADER
#NOTE TO GRADER

import sys,math,copy

CRITICAL_VALUE = 6.635

class Data:
	def __init__(self):
		#label really means class
		self.label = 0
		self.attributes = dict()

	def setLabel(self,l):
		self.label = l

	def setAttribute(self,attr,value):
		self.attributes[attr] = value

class Node:
	def __init__(self,parent):
		self.classifier = None
		self.parent = parent
		self.clas = None
		self.trueChild = None
		self.falseChild = None

	def setClassifier(self,attr):
		self.classifier = attr

	def setTrueChild(self,child):
		self.trueChild = child

	def setFalseChild(self,child):
		self.falseChild = child

	def setClas(self,clas):
		self.clas = clas

class Tree:
	def __init__(self,training,test,model):
		data = self.read_csv(training)
		self.root = Node(self)
		self.build_tree(self.root,data)
		accuracy = self.test(test)
		print("Accuracy is " + str(accuracy))
		self.export_model(model)

	def draw_tree(self):
		print("root", self.root.classifier)
		print("false child", self.root.falseChild.classifier)
		print("true child", self.root.trueChild.classifier)


	def build_tree(self,node,data):
		gain = dict()
		for attr in data[0].attributes:
			gain[attr] = 0;
		if len(gain) == 0:
			#base case, no attributes left to split on
			self.predict(node,data)
			return

		for attr in gain:
			gain[attr] = self.calc_informationgain(copy.deepcopy(data),attr)

		curr = 0
		currattr = ''
		for attr in gain:
			if gain[attr] >= curr:
				currattr = attr
				curr = gain[attr]

		node.setClassifier(currattr)
		node.setTrueChild(Node(self))
		node.setFalseChild(Node(self))

		#if (self.should_stop(data,currattr) == 1):
		if curr == 0:
			self.predict(node,data)
			return

		#do I also need to stop if info gain is 0?

		truelist, falselist = self.strip_data(data,node.classifier)

		if len(truelist) == 0:
			node.setClas(0)
		else:
			self.build_tree(node.trueChild,truelist)

		if len(falselist) == 0:
			node.setClas(1)
		else:
			self.build_tree(node.falseChild,falselist)


	def strip_data(self,data, attr):
		truelist = []
		falselist = []
		for dataobj in data:
			if (attr not in dataobj.attributes):
				continue

			if dataobj.attributes[attr] == True:
				del dataobj.attributes[attr]
				truelist.append(dataobj)
			else:
				del dataobj.attributes[attr]

				falselist.append(dataobj)
		return truelist, falselist

	def should_stop(self,data,attr):
		#returns True or False based on the results of the chi squared test
		p = 0.0
		n = 0.0
		p1=0.0
		p2=0.0
		n1=0.0
		n2=0.0
		for r in data:
			if r.label == 1:
				p+=1
			else:
				n+=1
			if r.attributes[attr] == 1:
				#that means it is c1
				if r.label == 1:
					p1+=1
				else:
					n1+=1
			else:
				#must be c2
				if r.label == 1:
					p2+=1
				else:
					n1+=1
		pp1 = p*((p1+n1)/len(data))
		pp2 = p*((p2+n2)/len(data))
		nn1 = n*((p1+n1)/len(data))
		nn2 = n*((p2+n2)/len(data))
		chi1 = (((p1-pp1)**2)/pp1)+(((n1-nn1)**2)/nn1)
		chi2 = (((p2-pp2)**2)/pp2)+(((n2-nn2)**2)/nn2)

		print(chi1+chi2)
		return ((chi1+chi2) > CRITICAL_VALUE)

	def calc_entropy(self,proportion):
		if (float(proportion) == 0) or (float(proportion) == 1):
			return 0
		return (-1*proportion*(math.log(float(proportion),2))) + (-1*(1-proportion)*(math.log(float(1-proportion),2)))

	def calc_informationgain(self,data,attr):
		truelist, falselist = self.strip_data(data,attr)
		proportion = float(len(truelist))/float(len(data))
		S = 0.0
		for r in data:
			if r.label == 1:
				S+= 1
		T = 0.0
		F = 0.0
		for r in truelist:
			if r.label == 1:
				T+=1
		for r in falselist:
			if r.label == 1:
				F+=1
		if (len(data) == 0) or (len(truelist) == 0) or (len(falselist) == 0):
			return 0
		return self.calc_entropy(float(S)/float(len(data))) - ( (proportion*self.calc_entropy(T/len(truelist))) + ((1 - proportion)*self.calc_entropy(F/len(falselist))) )

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
					if (attr == 'Class'):
						continue
					attrs.append(attr)
				continue
			dataobj = Data()
			for i in range(len(attrs)):
				dataobj.setAttribute(attrs[i],int(l[i]))
			dataobj.setLabel(int(l[i+1]))
			data.append(dataobj)
		f.close()
		return data

	def predict(self,node,data):
		positive = 0
		negative = 0
		for obj in data:
			if obj.label == True:
				positive += 1
			else:
				negative += 1
		node.clas = (positive >= negative)

	def predictanswer(self,question,node):
		if node.clas != None:
			return node.clas
		if question.attributes[node.classifier] == 1:
			return self.predictanswer(question,node.trueChild)
		else:
			return self.predictanswer(question,node.falseChild)

	def test(self,filename):
		testdata = self.read_csv(filename)
		total = 0
		correct = 0
		for question in testdata:
			answer = self.predictanswer(question,self.root)
			total += 1
			correct += (answer == question.label)
		return float(correct)/float(total)

	def export_model(self,filename):
		f = open(filename,"w")
		self.writenode(f,self.root,0)
		f.close()
		return

	def writenode(self,f,node, amt):
		for i in range(amt):
			f.write("| ")
		f.write(node.classifier)
		f.write(" = 0 : ")
		if (node.falseChild.clas != None):
			f.write(str(node.falseChild.clas))
		else:
			f.write("\n")
			self.writenode(f,node.falseChild,amt+1)
		f.write("\n")
		for i in range(amt):
			f.write("| ")
		f.write(node.classifier)
		f.write(" = 1 : ")
		if (node.trueChild.clas != None):

			f.write(str(node.trueChild.clas))
		else:
			f.write("\n")
			self.writenode(f,node.trueChild,amt+1)
		return

def main():
	tree = Tree(sys.argv[1],sys.argv[2],sys.argv[3])


main()
