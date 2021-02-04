import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
import sklearn.linear_model as lm
import sklearn.tree
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
warnings.simplefilter("ignore")




df = pd.read_csv('pre-processed data.csv')
bdf=df

le = preprocessing.LabelEncoder()
df=df.apply(le.fit_transform)





#xls = pd.ExcelFile('oasis_longitudinal.csv')
#df1 = pd.read_excel(xls, 0)
#df2 = pd.read_excel(xls, 1)

#datatrain = df1.to_numpy()
#datatest = df2.to_numpy()



#2,5,6 need to be labelencoded


def predsvm(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = SVC(kernel='linear')
	classifier.fit(X_train,y_train)

	print('SVM linear kernel for '+str(feat)+' %test:'+str(perctst))
	predicted=classifier.predict(X_test)

	#print('predicted:',predicted)
	#print('actual:',y_test)

	right=0
	wrong=0
	for i in range(len(predicted)):
		if predicted[i]==y_test[i]:
			right+=1
		else:
			wrong+=1

	print('accuracy:',right/(right+wrong))

def predrandfor(X_train,X_test,y_train,y_test,perctst,feat):


	#X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
	classifier = RandomForestClassifier(max_depth=4, random_state=0)
	classifier.fit(X_train,y_train)

	print('RandomForest for '+str(feat)+' %test:'+str(perctst))
	predicted=classifier.predict(X_test)


	right=0
	wrong=0
	for i in range(len(predicted)):
		if predicted[i]==y_test[i]:
			right+=1
		else:
			wrong+=1

	print('accuracy:',right/(right+wrong))

def predlogreg(X_train,X_test,y_train,y_test,perctst,feat):


	#X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
	lr = lm.LogisticRegression()
	lr.fit(X_train,y_train)

	print('LogisticRegression for '+str(feat)+' %test:'+str(perctst))
	predicted=lr.predict(X_test)


	right=0
	wrong=0
	for i in range(len(predicted)):
		if predicted[i]==y_test[i]:
			right+=1
		else:
			wrong+=1

	print('accuracy:',right/(right+wrong))

def preddectree(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = sklearn.tree.DecisionTreeClassifier()
	classifier.fit(X_train,y_train)
	print('DecisionTree for '+str(feat)+' %test:'+str(perctst))
	predicted=classifier.predict(X_test)


	right=0
	wrong=0
	for i in range(len(predicted)):
		if predicted[i]==y_test[i]:
			right+=1
		else:
			wrong+=1

	print('accuracy:',right/(right+wrong))




def predsvmroc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = OneVsRestClassifier(SVC(kernel='linear'))
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
	    plt.plot(fpr[i], tpr[i], label='ROC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, roc_auc[i]))


	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('LinearSVM ROC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()

def predrandforroc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = RandomForestClassifier(max_depth=4, random_state=0)
	classifier.fit(X_train, y_train)
	y_score = classifier.predict(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
	    plt.plot(fpr[i], tpr[i], label='ROC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, roc_auc[i]))


	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('RandomForest ROC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()


def predlogregroc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = OneVsRestClassifier(lm.LogisticRegression())
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
	    plt.plot(fpr[i], tpr[i], label='ROC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, roc_auc[i]))


	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('LogisticRegression ROC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()


def preddectreeroc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = sklearn.tree.DecisionTreeClassifier()
	classifier.fit(X_train, y_train)
	y_score = classifier.predict(X_test)
	#classifier = OneVsRestClassifier(sklearn.tree.DecisionTreeClassifier())
	
	#y_score = classifier.fit(X_train, y_train).decision_function(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
	    plt.plot(fpr[i], tpr[i], label='ROC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, roc_auc[i]))


	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('DecisionTree ROC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()

def predsvmprc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = OneVsRestClassifier(SVC(kernel='linear',probability=True))
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)



	

	lr_probs = classifier.predict_proba(X_test)
	#print(lr_probs[:,1][0])
	#print(y_test[0])
	# keep probabilities for the positive outcome only
	#lr_probs = lr_probs[:, 1]
	# predict class values
	yhat = classifier.predict(X_test)
	#lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	#lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
	# summarize scores
	lr_prec = dict()
	lr_rec = dict()
	prc_auc = dict()
	prc_f1 = dict()
	for i in range(n_classes):
		lr_prec[i], lr_rec[i], _ = precision_recall_curve(y_test[:, i], lr_probs[:, i])
		prc_auc[i] = average_precision_score(y_test[:, i], lr_probs[:, i])
		#temptuplst=[]
		#for n in range(len(lr_rec[i])):
		#	temptuplst.append((lr_rec[i][n],lr_prec[i][n]))
		#sortedlist=sorted(temptuplst,key=lambda x:x[0])
		#newlr_rec=[x[0] for x in sortedlist]
		#newlr_prec=[x[1] for x in sortedlist]
		#prc_auc[i] = auc(lr_prec[i], lr_rec[i])
		#print('did it')
	#	prc_f1,prc_auc[i] = f1_score(y_test[:,i], yhat[:,i]),auc(lr_prec[i], lr_rec[i])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}')
		
		plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, prc_auc[i]))
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}'
		#		           ''.format(i))


	#print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
	# plot the precision-recall curves
	no_skill = len(y_test[y_test==1]) / len(y_test)
	#plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.0])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('LinearSVM PRC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()


	

def predrandforprc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = RandomForestClassifier(max_depth=4, random_state=0)
	#y_score = classifier.fit(X_train, y_train).decision_function(X_test)
	
	classifier.fit(X_train, y_train)	

	lr_probs = classifier.predict_proba(X_test)
	print(lr_probs[0])
	#BINARIZE IT HERE
	y_test=label_binarize(y_test, classes=[0,1,2,3])
	print(y_test[0])


	#print(y_test)
	#print(lr_probs[:,1][0])
	#print(y_test[0])
	# keep probabilities for the positive outcome only
	#lr_probs = lr_probs[:, 1]
	# predict class values
	yhat = classifier.predict(X_test)
	#lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	#lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
	# summarize scores
	lr_prec = dict()
	lr_rec = dict()
	prc_auc = dict()
	prc_f1 = dict()
	for i in range(n_classes):
		lr_prec[i], lr_rec[i], _ = precision_recall_curve(y_test[:, i], lr_probs[:, i])
		prc_auc[i] = average_precision_score(y_test[:, i], lr_probs[:, i])
		#temptuplst=[]
		#for n in range(len(lr_rec[i])):
		#	temptuplst.append((lr_rec[i][n],lr_prec[i][n]))
		#sortedlist=sorted(temptuplst,key=lambda x:x[0])
		#newlr_rec=[x[0] for x in sortedlist]
		#newlr_prec=[x[1] for x in sortedlist]
		#prc_auc[i] = auc(lr_prec[i], lr_rec[i])
		#print('did it')
	#	prc_f1,prc_auc[i] = f1_score(y_test[:,i], yhat[:,i]),auc(lr_prec[i], lr_rec[i])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}')
		
		plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, prc_auc[i]))
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}'
		#		           ''.format(i))


	#print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
	# plot the precision-recall curves
	no_skill = len(y_test[y_test==1]) / len(y_test)
	#plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.0])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('RandomForest PRC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()

def predlogregprc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = OneVsRestClassifier(lm.LogisticRegression())
	y_score = classifier.fit(X_train, y_train).decision_function(X_test)
	

	lr_probs = classifier.predict_proba(X_test)
	#print(lr_probs[:,1][0])
	#print(y_test[0])
	# keep probabilities for the positive outcome only
	#lr_probs = lr_probs[:, 1]
	# predict class values
	yhat = classifier.predict(X_test)
	#lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	#lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
	# summarize scores
	lr_prec = dict()
	lr_rec = dict()
	prc_auc = dict()
	prc_f1 = dict()
	for i in range(n_classes):
		lr_prec[i], lr_rec[i], _ = precision_recall_curve(y_test[:, i], lr_probs[:, i])
		prc_auc[i] = average_precision_score(y_test[:, i], lr_probs[:, i])
		#temptuplst=[]
		#for n in range(len(lr_rec[i])):
		#	temptuplst.append((lr_rec[i][n],lr_prec[i][n]))
		#sortedlist=sorted(temptuplst,key=lambda x:x[0])
		#newlr_rec=[x[0] for x in sortedlist]
		#newlr_prec=[x[1] for x in sortedlist]
		#prc_auc[i] = auc(lr_prec[i], lr_rec[i])
		#print('did it')
	#	prc_f1,prc_auc[i] = f1_score(y_test[:,i], yhat[:,i]),auc(lr_prec[i], lr_rec[i])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}')
		
		plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, prc_auc[i]))
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}'
		#		           ''.format(i))


	#print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
	# plot the precision-recall curves
	no_skill = len(y_test[y_test==1]) / len(y_test)
	#plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.0])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('LogisticRegression PRC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()


def preddectreeprc(X_train,X_test,y_train,y_test,perctst,feat):
	classifier = sklearn.tree.DecisionTreeClassifier()
	classifier.fit(X_train, y_train)
	#y_score = classifier.fit(X_train, y_train).decision_function(X_test)
	

	lr_probs = classifier.predict_proba(X_test)
	y_test=label_binarize(y_test, classes=[0,1,2,3])
	print(lr_probs[0])
	print(y_test[0])
	#print(lr_probs[:,1][0])
	#print(y_test[0])
	# keep probabilities for the positive outcome only
	#lr_probs = lr_probs[:, 1]
	# predict class values
	yhat = classifier.predict(X_test)
	#lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
	#lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
	# summarize scores
	lr_prec = dict()
	lr_rec = dict()
	prc_auc = dict()
	prc_f1 = dict()
	for i in range(n_classes):
		lr_prec[i], lr_rec[i], _ = precision_recall_curve(y_test[:, i], lr_probs[:, i])
		prc_auc[i] = average_precision_score(y_test[:, i], lr_probs[:, i])
		#temptuplst=[]
		#for n in range(len(lr_rec[i])):
		#	temptuplst.append((lr_rec[i][n],lr_prec[i][n]))
		#sortedlist=sorted(temptuplst,key=lambda x:x[0])
		#newlr_rec=[x[0] for x in sortedlist]
		#newlr_prec=[x[1] for x in sortedlist]
		#prc_auc[i] = auc(lr_prec[i], lr_rec[i])
		#print('did it')
	#	prc_f1,prc_auc[i] = f1_score(y_test[:,i], yhat[:,i]),auc(lr_prec[i], lr_rec[i])



	# Plot ROC curve
	plt.figure()
	for i in range(n_classes):
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}')
		
		plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0} (area = {1:0.2f})'
				           ''.format(i, prc_auc[i]))
		#plt.plot(lr_rec[i], lr_prec[i], label='PRC for '+str(feat)+' of class {0}'
		#		           ''.format(i))


	#print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
	# plot the precision-recall curves
	no_skill = len(y_test[y_test==1]) / len(y_test)
	#plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.0])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('DecisionTree PRC for predicting multiple classes of '+str(feat)+' %test:'+str(perctst*100))
	plt.legend(loc="lower right")
	plt.show()
#data = df.to_numpy()
#print(data[:,2])
data=df

for perctst in [.15]:
	print('-----------------------------------------------------------')
	for feat in ['CDR']:

		actlist=['Group','Visit','MR Delay','M/F','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF']
		featofintrst=feat
		currxcol=[x for x in actlist if x != featofintrst]
		#print(currxcol)
		#print(data[currxcol[0]].values)


		X = data[currxcol].values
		y = data[featofintrst].values

		
		#print(len(X))


		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perctst, random_state=42)

		#JUST FOR ACCURACY
		predsvm(X_train,X_test,y_train,y_test,perctst*100,feat)
		predrandfor(X_train,X_test,y_train,y_test,perctst*100,feat)
		predlogreg(X_train,X_test,y_train,y_test,perctst*100,feat)
		preddectree(X_train,X_test,y_train,y_test,perctst*100,feat)


		X = data[currxcol].values
		y = data[featofintrst].values

		y = label_binarize(y, classes=[0,1,2,3])
		n_classes = y.shape[1]

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perctst, random_state=42)


		'''
		#ROC CURVES
		predsvmroc(X_train,X_test,y_train,y_test,perctst,feat)
		predrandforroc(X_train,X_test,y_train,y_test,perctst,feat)
		predlogregroc(X_train,X_test,y_train,y_test,perctst,feat)
		preddectreeroc(X_train,X_test,y_train,y_test,perctst,feat)
		'''


		#PRC CURVES
		predsvmprc(X_train,X_test,y_train,y_test,perctst,feat)

		X = data[currxcol].values
		y = data[featofintrst].values
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perctst, random_state=42)

		predrandforprc(X_train,X_test,y_train,y_test,perctst,feat)
	
		X = data[currxcol].values
		y = data[featofintrst].values
		y = label_binarize(y, classes=[0,1,2,3])
		n_classes = y.shape[1]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perctst, random_state=42)

		predlogregprc(X_train,X_test,y_train,y_test,perctst,feat)

		X = data[currxcol].values
		y = data[featofintrst].values
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=perctst, random_state=42)

		preddectreeprc(X_train,X_test,y_train,y_test,perctst,feat)
		









