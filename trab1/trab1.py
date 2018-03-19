#!/usr/bin/env python3
import sys
import matplotlib.pyplot as pyplot

class Data:
	def __init__(self,infile):
		with open(infile) as f:
			data = f.read().split()
		self.X = []
		for i in data:
			self.X.append(list(map(float,i.split(',')[:4])))

	def show(self):
		for i in self.X:
			print(i)

	def plot(self):
		pyplot.plot(self.X)
		pyplot.show()


def main(infile='iris.data.txt'):
	matrix = Data(infile)
	# matrix.show()
	matrix.plot()

if(__name__ == '__main__'):
	if(len(sys.argv) > 1):
		main(sys.argv[1])
	else:
		main()
