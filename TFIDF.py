import csv
from math import log
from difflib import SequenceMatcher


class TFIDF:
    def __init__(self,trainFile, testFile, datafile1,datafile2):
        print("Using trainFile to construct titleItemCount")
        datafile=open(trainFile)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #initialize maps
        #map of each item with a map of words and counts
        self.titleItemCount={}
        self.D1ItemCount={}
        self.D2ItemCount={}
        
        #record total occurrences of each word among all items
        self.titleTotalCount={}
        self.D1TotalCount={}
        self.D2TotalCount={}
        
        #keep track of total number of items for each dictionary
        self.titleTotalItems=0
        self.D1TotalItems=0
        self.D2TotalItems=0
        
        #skip first line
        dataReader.next()  
        #process file
        for info in dataReader:
            itemID=info[1]
            #only use information if item title was not previously add
            #this is to avoid repetition
            if(not itemID in self.titleItemCount):
                attributes=[]
                for i in range(2,len(info)-2):
                    temp=info[i].split(" ")
                    for item in temp:
                        attributes.append(item.strip(',').strip('.').strip('(').strip(')').strip('"'))
                #initialize new map if itemCount does not contain current item
                tempMap={}
                self.titleTotalItems+=1
                for item in attributes:
                    #update information in itemCount
                    if not item.lower() in tempMap:
                        tempMap[item.lower()]=1
                    else:
                        tempMap[item.lower()]+=1
                self.titleItemCount[itemID]=tempMap
        datafile.close()
        print("Done with training data.")  
        
        print("Using testFile to construct titleItemCount")
        datafile=open(testFile)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #skip first line
        dataReader.next()  
        #process file
        for info in dataReader:
            itemID=info[1]
            #only use information if item title was not previously add
            #this is to avoid repetition
            if(not itemID in self.titleItemCount):
                attributes=[]
                for i in range(2,len(info)-1):
                    temp=info[i].split(" ")
                    for item in temp:
                        attributes.append(item.strip(',').strip('.').strip('(').strip(')').strip('"'))
                #initialize new map if itemCount does not contain current item
                tempMap={}
                self.titleTotalItems+=1
                for item in attributes:
                    #update information in itemCount
                    if not item.lower() in tempMap:
                        tempMap[item.lower()]=1
                    else:
                        tempMap[item.lower()]+=1
                self.titleItemCount[itemID]=tempMap
        print("Done with training data.")  
        datafile.close()
        
        print("Using first data file to construct D1ItemCount")
        #run through first data file
        datafile=open(datafile1)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        dataReader.next()
        #process file
        for info in dataReader:
            itemID=info[0]
            attributes=[]
            for i in range(1,len(info)):
                temp=info[i].split(" ")
                for item in temp:
                    attributes.append(item.strip(',').strip('.').strip('(').strip(')').strip('"'))
            #initialize new map if itemCount does not contain current item
            if not itemID in self.D1ItemCount:
                tempMap={}
                self.D1TotalItems+=1
            #else retrieve map
            else:
                tempMap=self.D1ItemCount[itemID]
            for item in attributes:
                #update information in itemCount
                if not item.lower() in tempMap:
                    tempMap[item.lower()]=1
                else:
                    tempMap[item.lower()]+=1
            self.D1ItemCount[itemID]=tempMap
             
        print("Done with first data file.")    
        datafile.close()
 
        print("Using second data file to construct D2ItemCount")
        #run through second data file
        datafile=open(datafile2)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        #skip first line
        dataReader.next()
        #process file
        for info in dataReader:
            itemID=info[0]
            attributes=[]
            for i in range(1,len(info)):
                temp=info[i].split(" ")
                for item in temp:
                    attributes.append(item.strip(',').strip('.').strip('(').strip(')').strip('"'))
            #initialize new map if itemCount does not contain current item
            if not itemID in self.D2ItemCount:
                tempMap={}
                self.D2TotalItems+=1
            #else retrieve map
            else:
                tempMap=self.D2ItemCount[itemID]
            for item in attributes:
                #update information in itemCount
                if not item.lower() in tempMap:
                    tempMap[item.lower()]=1
                else:
                    tempMap[item.lower()]+=1
            self.D2ItemCount[itemID]=tempMap
        
        print("Done with second data file.")  
        datafile.close()
        
        print("Constructing totalCount") 
        #update totalCount
        itemset=self.titleItemCount.keys()
        for item in itemset:
            wordset=self.titleItemCount[item].keys()
            for word in wordset:
                if not word.lower() in self.titleTotalCount:
                    self.titleTotalCount[word.lower()]=1
                else:
                    self.titleTotalCount[word.lower()]+=1
        
        #update totalCount
        itemset=self.D1ItemCount.keys()
        for item in itemset:
            wordset=self.D1ItemCount[item].keys()
            for word in wordset:
                if not word.lower() in self.D1TotalCount:
                    self.D1TotalCount[word.lower()]=1
                else:
                    self.D1TotalCount[word.lower()]+=1
                    
        #update totalCount
        itemset=self.D2ItemCount.keys()
        for item in itemset:
            wordset=self.D2ItemCount[item].keys()
            for word in wordset:
                if not word.lower() in self.D2TotalCount:
                    self.D2TotalCount[word.lower()]=1
                else:
                    self.D2TotalCount[word.lower()]+=1
        print("Constructing totalCount complete.") 
    
    #returns similiarity between strings
    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    def getRelevance(self,item,keywords):
        titleScore=0
        D1Score=0
        D2Score=0
        searchwords=keywords.split(" ")
        if item in self.titleItemCount:
            for word in searchwords:
                keyset=self.titleItemCount[item].keys()
                for key in keyset:
                    eval=self.similar(key, word.lower())
                    if eval>0.5:
                        idf=log(self.titleTotalItems/self.titleTotalCount[key])
                        titleScore+=self.titleItemCount[item][key]*idf*eval 
        if item in self.D1ItemCount:
            for word in searchwords:
                keyset=self.D1ItemCount[item].keys()
                for key in keyset:
                    eval=self.similar(key, word.lower())
                    if eval>0.5:
                        idf=log(self.D1TotalItems/self.D1TotalCount[key])
                        D1Score+=self.D1ItemCount[item][key]*idf*eval
        if item in self.D2ItemCount:
            for word in searchwords:
                keyset=self.D2ItemCount[item].keys()
                for key in keyset:
                    eval=self.similar(key, word.lower())
                    if eval>0.5:
                        idf=log(self.D2TotalItems/self.D2TotalCount[key])
                        D2Score+=self.D2ItemCount[item][key]*idf*eval
        return (titleScore/len(searchwords),D1Score/len(searchwords),D2Score/len(searchwords))
    
    #returns item ID, generated data and original relevance
    def constructTestData(self, trainFile):
        datafile=open(trainFile)
        dataReader = csv.reader(datafile, delimiter=',', quotechar='|')
        items=[]
        relevance=[]
        titleGeneratedData=[]
        D1GeneratedData=[]
        D2GeneratedData=[]
        dataReader.next();
        for info in dataReader:
            infoSize=len(info)
            itemID=info[1]
            items.append(itemID)
            relevance.append(info[infoSize-1])
            (titleData,D1Data,D2Data)=self.getRelevance(itemID, info[infoSize-2].strip('"'))
            titleGeneratedData.append(titleData)
            D1GeneratedData.append(D1Data)
            D2GeneratedData.append(D2Data)
            print([itemID,titleData,D1Data,D2Data,info[infoSize-1]])
        return (items,titleGeneratedData,D1GeneratedData,D2GeneratedData,relevance)

newSearch=TFIDF("train.csv","test.csv","product_descriptions.csv","attributes.csv")
newSearch.constructTestData("train.csv")
# 
# while(True):
#     item=raw_input("Enter item name: ")
#     keyword=raw_input("Enter search phrases: ")
#     print(newSearch.getRelevance(item, keyword))