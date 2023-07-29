#  DataSet Download Link 
#  --->  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
import pandas as panda
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# loading the downloaded credit card data to the pandas dataframe
creditCardData = panda.read_csv('Credit Card Dataset/creditcard.csv')

# printing first five row of data collected
# print(creditCardData.head())


# data information
# print(creditCardData.info())

# getting distribution of legal transactions & fraudulent transactions
# print(creditCardData['Class'].value_counts())

# separating the data for analysis
legal = creditCardData[creditCardData.Class == 0]
fraud = creditCardData[creditCardData.Class == 1]

# printing the acquired legal and fraud data
# print(legal)
# print(fraud)

# print dimensions
# print(legal.shape)
# print(fraud.shape)

# statistical measures of legal data
# print(legal.Amount.describe())

# statistical measures of fraud data
# print(fraud.Amount.describe())

# comparing the values for legal and fraud transactions
# print(creditCardData.groupby('Class').mean())

# In our dataset we have 492 fraudulent transactions
# Taking 492 legal smples as well
legalSample = legal.sample(n=492)


# Concatinating the collected sample of datdframe and fraud dataframe
newData = panda.concat([legalSample, fraud], axis=0)

# Printing new data
# print(newData)

# First five rows
# newData.head()

# Printing the legal and fraudulent data count of new data.
# print(newData['Class'].value_counts())

# splitting the data into features and targets
feature = newData.drop(columns='Class', axis=1)
target= newData['Class']

# print(feature)
# print(target)

# Now splitting the feature and target into trainig and testig data
featureTrain, featureTest, targetTrain, targetTest = train_test_split(feature, target, test_size=0.2, stratify=target, random_state=2)


component=LogisticRegression(max_iter=1000)

# training the logistic regression model with  training data collected
component.fit(featureTrain,targetTrain)

# accuracy on training data
featureTrainPrediction = component.predict(featureTrain)
trainingAccuracy = accuracy_score(featureTrainPrediction, targetTrain)
trainingF1 = f1_score(featureTrainPrediction, targetTrain)
trainingPrecision = precision_score(featureTrainPrediction, targetTrain)
trainingRecall = recall_score(featureTrainPrediction, targetTrain)


# printing accuracy
print("Accuracy of training data: " , trainingAccuracy)
print("F1 score on training data: " , trainingF1)
print("Precision on training data:", trainingPrecision)
print("Recall on training data:", trainingRecall)


# accuracy on test data
featureTestPrediction = component.predict(featureTest)
testAccuracy = accuracy_score(featureTestPrediction, targetTest)
testF1 = f1_score(featureTestPrediction, targetTest)
testPrecision = precision_score(featureTestPrediction, targetTest)
testRecall = recall_score(featureTestPrediction, targetTest)

# printing accuracy
print("Accuracy of test data: " , testAccuracy)
print("F1 score on test data: " , testF1)
print("Precision on test data:", testPrecision)
print("Recall on test data:", testRecall)




# class Solution {
#     public int alternatingSubarray(int[] nums) {
#         int count[]=new int[nums.length-1];
#         for(int i=1;i<nums.length;i++)
#         {
#             count[i-1]=nums[i]-nums[i-1];
#         }
#         int max=0;
#         int help=0;
#         int need=1;
#         for(int i=0;i<count.length;i++)
#         {
#             if(count[i]==need)
#             {
#                 help++;
#                 need=-need;
#                 max=Math.max(max,help);
#             }
#             else if(count[i]==1){
#                 help=1;
#                 need=-1;
#             }
#             else{
#                 help=0;
#                 need=1;
#             }
#         }
#         if(max==0) return -1;
#         return max+1;
#     }
# }