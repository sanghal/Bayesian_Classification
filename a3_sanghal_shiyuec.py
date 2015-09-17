import sys

TEST_IMAGE_FILE = sys.argv[1]
TEST_LABEL_FILE = sys.argv[2]

cl="cl"

from trainingclass import *
from pprint import *

def train():
  global cl
  print("Loading training images and labels...")
  train_img_list= readImages("trainingimages.txt",5000)
  train_label_list = readLabels("traininglabels.txt",5000)
  print("Training images and labels loaded")

  cl=naivebayes(getfeatures)
  for i in range(len(train_img_list)):
  	cl.train(train_img_list[i],train_label_list[i])
  print("Classifier trained")

def test(cl):
	print("Start Testing...")
	test_img_list= readImages(TEST_IMAGE_FILE,1000)
	test_label_list = readLabels(TEST_LABEL_FILE,1000)

	conf_matrix = generate_matrix()
	percent_matrix = generate_matrix()
  
	for i in range(len(test_img_list)):
		cat = int(cl.classify(test_img_list[i],default='unknown')) # category classified
		label = int(test_label_list[i]) # get label
		conf_matrix[cat][label] +=1

		if i % 100 == 0 and i != 0 :
			print ("Finished " + str(int(i/100)) + "00 Images")

	
	for i in range(10):
		row_sum = rowsum(conf_matrix,i)
		for j in range(10):
			percent_matrix[i][j] = "{0:.0f}%".format(float(conf_matrix[i][j])/row_sum * 100)

	print("Performance results")
	pprint(conf_matrix)
	print("\n")
	pprint(percent_matrix)
	

def rowsum(matrix,row):
	rowsum = 0
	for i in range(len(matrix[row])):
		rowsum += matrix[row][i]
	return rowsum

def generate_matrix():
	matrix = []
	for i in range(10):
		matrix.append([0 for i in range(10)])
	return matrix

train()
test(cl)



