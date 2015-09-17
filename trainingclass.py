import re
import math


def readImages(fname,numImages):
  image_height = 28
  f = open(fname,"rU")
  images = []
  for i in range(numImages):
    image = []
    for j in range(image_height): 
      image.append(f.readline().rstrip('\n'))
    images.append(image)
  return images


def readLabels(fname,numLabels):
  f = open(fname,"rU")
  labels = []
  for i in range(numLabels): 
    labels.append(f.readline().rstrip('\n'))
  return labels

def getfeatures(img):
  img_height = 28
  img_width = 28
  full =[]
  for i in range(img_height):
    for j in range(img_width):
      part = (i,j,img[i][j])
      full.append(part)
  return full

class classifier:
  def __init__(self,getfeatures,filename=None):
    # Counts of feature/category combinations
    self.fc={}
    # Counts of images for 
    self.cc={}
    self.getfeatures=getfeatures
    
  # Increase the count of a feature/category pair
  def incf(self,f,cat):
    self.fc.setdefault(f,{})
    self.fc[f].setdefault(cat,0)
    self.fc[f][cat]+=1

  # Increase the count of a category
  def incc(self,cat):
    self.cc.setdefault(cat,0)
    self.cc[cat]+=1
 
  # The number of times a feature has appeared in a category
  def fcount(self,f,cat):
    if f in self.fc and cat in self.fc[f]: 

      return float(self.fc[f][cat])
    return 0.0
  
  # The number of items in a category
  def catcount(self,cat):
    if cat in self.cc:
      return float(self.cc[cat])
    return 0
  
  # The total number of items
  def totalcount(self):
    return sum(self.cc.values())

  # The list of all categories
  def categories(self):
    return self.cc.keys()

  def train(self,item,cat):
    features=self.getfeatures(item)
    # Increment the count for every feature with this category
    for f in features:  
      self.incf(f,cat)
    # Increment the count for this category
    self.incc(cat)
  
  def fprob(self,f,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.fcount(f,cat)/self.catcount(cat)

  def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
    # Calculate current probability
    basicprob=prf(f,cat)

    # Count the number of times this feature has appeared in
    # all categories
    totals=sum([self.fcount(f,c) for c in self.categories()])

    # Calculate the weighted average
    bp=((weight*ap)+(totals*basicprob))/(weight+totals)
    return bp


class naivebayes(classifier):
  
  def __init__(self,xfeatures):
    classifier.__init__(self,getfeatures)
    self.thresholds={}
  
  def docprob(self,item,cat):
    features=self.getfeatures(item)   

    # Multiply the probabilities of all the features together
    p=1
    for f in features: p*=self.weightedprob(f,cat,self.fprob)
    return p

  def prob(self,item,cat):
    catprob=self.catcount(cat)/self.totalcount()
    docprob=self.docprob(item,cat)
    return docprob*catprob
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]
  
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    max=0.0
    for cat in self.categories():
      probs[cat]=self.prob(item,cat)
      if probs[cat]>max: 
        max=probs[cat]
        best=cat

    # Make sure the probability exceeds threshold*next best
    for cat in probs:
      if cat==best: continue
      if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best

 
