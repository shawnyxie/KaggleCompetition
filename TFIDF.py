import csv
from math import log

class TFIDF:
    def __init__(self,trainFile, testFile, datafile1,datafile2):
        print("Using trainFile to construct itemCount")
        datafile=open(trainFile)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #initialize maps
        #map of each item with a map of words and counts
        self.itemCount={}
        #record total occurrences of each word among all items
        self.totalCount={}
        self.totalItems=0
        #skip first line
        temp=dataReader.next()  
        #process file
        for info in dataReader:
            itemID=info[1]
            #only use information if item title was not previously add
            #this is to avoid repetition
            if(not itemID in self.itemCount):
                attributes=[]
                for i in range(2,len(info)-2):
                    temp=info[i].split(" ")
                    for item in temp:
                        attributes.append(item.strip(','))
                #initialize new map if itemCount does not contain current item
                tempMap={}
                self.totalItems+=1
                for item in attributes:
                    #update information in itemCount
                    if not item in tempMap:
                        tempMap[item.lower()]=1
                    else:
                        tempMap[item.lower()]+=1
                self.itemCount[itemID]=tempMap
        datafile.close()
        print("Done with training data.")  
        
        print("Using testFile to construct itemCount")
        datafile=open(testFile)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #skip first line
        temp=dataReader.next()  
        #process file
        for info in dataReader:
            itemID=info[1]
            #only use information if item title was not previously add
            #this is to avoid repetition
            if(not itemID in self.itemCount):
                attributes=[]
                for i in range(2,len(info)-1):
                    temp=info[i].split(" ")
                    for item in temp:
                        attributes.append(item.strip(','))
                #initialize new map if itemCount does not contain current item
                tempMap={}
                self.totalItems+=1
                for item in attributes:
                    #update information in itemCount
                    if not item in tempMap:
                        tempMap[item.lower()]=1
                    else:
                        tempMap[item.lower()]+=1
                self.itemCount[itemID]=tempMap
        print("Done with training data.")  
        datafile.close()
        
        print("Using first data file to construct itemCount")
        #run through first data file
        datafile=open(datafile1)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #process file
        for info in dataReader:
            itemID=info[0]
            attributes=[]
            for i in range(1,len(info)):
                temp=info[i].split(" ")
                for item in temp:
                    attributes.append(item.strip(','))
            #initialize new map if itemCount does not contain current item
            if not itemID in self.itemCount:
                tempMap={}
                self.totalItems+=1
            #else retrieve map
            else:
                tempMap=self.itemCount[itemID]
            for item in attributes:
                #update information in itemCount
                if not item in tempMap:
                    tempMap[item.lower()]=1
                else:
                    tempMap[item.lower()]+=1
            self.itemCount[itemID]=tempMap
             
        print("Done with first data file.")    
        datafile.close()
 
        print("Using second data file to construct itemCount")
        #run through second data file
        datafile=open(datafile2)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #skip first line
        temp=dataReader.next()
        #process file
        for info in dataReader:
            itemID=info[0]
            attributes=[]
            for i in range(1,len(info)):
                temp=info[i].split(" ")
                for item in temp:
                    attributes.append(item.strip(','))
            #initialize new map if itemCount does not contain current item
            if not itemID in self.itemCount:
                tempMap={}
                self.totalItems+=1
            #else retrieve map
            else:
                tempMap=self.itemCount[itemID]
            for item in attributes:
                #update information in itemCount
                if not item in tempMap:
                    tempMap[item.lower()]=1
                else:
                    tempMap[item.lower()]+=1
            self.itemCount[itemID]=tempMap
        
        print("Done with second data file.")  
        datafile.close()
        
        print("Constructing totalCount") 
        #update totalCount
        itemset=self.itemCount.keys()
        for item in itemset:
            wordset=self.itemCount[item].keys()
            for word in wordset:
                if not word in self.totalCount:
                    self.totalCount[word.lower()]=1
                else:
                    self.totalCount[word.lower()]+=1

        print("Constructing totalCount complete.") 

    def getRelevance(self,item,keywords):
        score=0
        searchwords=keywords.split(" ")
        if item in self.itemCount:
            for word in searchwords:
                if word.lower() in self.itemCount[item]:
                    temp=self.totalCount[word.lower()]
                    idf=log(self.totalItems/temp)
                    newtemp=self.itemCount[item][word.lower()]
                    score+=newtemp*idf
        return score/len(searchwords)
    
    #returns item ID, generated data and original relevance
    def constructTestData(self, trainFile):
        datafile=open(trainFile)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        items=[]
        relevance=[]
        generatedData=[]
        dataReader.next();
        for info in dataReader:
            infoSize=len(info)
            itemID=info[1]
            items.append(itemID)
            relevance.append(info[infoSize-1])
            newData=self.getRelevance(itemID, info[infoSize-2])
            generatedData.append(newData)
            print([itemID,newData,info[infoSize-1]])
        #return (items,generatedData,relevance)

newSearch=TFIDF("train.csv","test.csv","product_descriptions.csv","attributes.csv")
newSearch.constructTestData("train.csv")
#  
# while(True):
#     item=raw_input("Enter item name: ")
#     keyword=raw_input("Enter search phrases: ")
#     print(newSearch.getRelevance(item, keyword))