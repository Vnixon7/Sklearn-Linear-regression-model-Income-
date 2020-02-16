import sklearn
from sklearn import linear_model,preprocessing
import pandas as pd
import numpy as np
import pickle

#uploading data
data = pd.read_csv('incomePredictionData.data')

#creating label for shift to integer data
label_maker = preprocessing.LabelEncoder()

#Transforming to Integer
age = label_maker.fit_transform(data['age'])
workclass = label_maker.fit_transform(data['workclass'])
finalweight = label_maker.fit_transform(data['finalweight'])
education = label_maker.fit_transform(data['education'])
education_num = label_maker.fit_transform(data['education-num'])
marital_status = label_maker.fit_transform(data['marital-status'])
occupation = label_maker.fit_transform(data['occupation'])
relationship = label_maker.fit_transform(data['relationship'])
race = label_maker.fit_transform(data['race'])
sex = label_maker.fit_transform(data['sex'])
capital_gain = label_maker.fit_transform(data['capital-gain'])
capital_loss = label_maker.fit_transform(data['capital-loss'])
hours_per_week = label_maker.fit_transform(data['hours-per-week'])
native_country = label_maker.fit_transform(data['native-country'])
moneys = label_maker.fit_transform(data['income'])

#prediction
predict = 'race'

#ziping every line of data
x = list(zip(age,workclass,finalweight,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,moneys))
y = list(race)

#initializing the train_test_split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


#Pickle saved training model to load back in
'''best = 1.0
for i in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('interation: ', i, 'Accuracy: ', acc * 100)

    if acc > best:
        best = acc
        with open('LR3_model.pickle', 'wb') as f:
            pickle.dump(linear, f)
            print(best)
            break'''

#loading best model to be used for prediction
load_in = open('LR3_model.pickle', 'rb')
linear = pickle.load(load_in)



#visual of co-efficent, and intercept
print('Co-efficent:', linear.coef_)
print('Intercept:', linear.intercept_)
label = 'blank'
label2 = 'blank'

#predicting the x
predictions = linear.predict(x_test)

#Changing the visual label
for i in range(len(predictions)):
    if predictions[i] >= 0.6 and predictions[i] <= 1.0:
        label = 'Amer-Indian-Eskimo'
    if predictions[i] >= 1.6 and predictions[i] <= 2.0:
        label = 'Asian-Pac-Islander'
    if predictions[i] >= 2.6 and predictions[i] <= 3.0:
        label = 'black'
    if predictions[i] >= 3.6:
        label = 'white'
    if y_test[i] >= 0.6 and predictions[i] <= 1.0:
        label2 = 'Amer-Indian-Eskimo'
    if y_test[i] >= 1.6 and predictions[i] <= 2.0:
        label2 = 'Asian-Pac-Islander'
    if y_test[i] >= 2.6 and predictions[i] <= 3.0:
        label2 = 'black'
    if y_test[i] >= 3.6:
        label2 = 'white'

    print('Prediction:',label,'Data:',x_test[i],'Actual:',label2)


'''#GRAPH VISUAL#
a = 'education'
style.use('ggplot')
pyplot.scatter(data[predict],data[a])
pyplot.xlabel('Race')
pyplot.ylabel('Education')
pyplot.show()'''
