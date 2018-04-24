import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import Vectorizer as vz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as all_score
import matplotlib.pyplot as plt

 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
train_filepath = root_dir+"/data/processed/basic_preprocessed.csv"
print("Train file:"+train_filepath)

trainData = pd.read_csv(train_filepath)
features = np.array(trainData.columns[3:])
ref_vect = vz.Vectorizer(trainData[:5], "comment_text", features)
print("target-dictionary")
target_labels = ref_vect.get_target_dict()
print(target_labels)
print("class vector:")
y = ref_vect.build_class_vector()
print(y)

print("term-document-matrix initial:")
print(ref_vect.get_term_document())

print("term-document-matrix after computation:")
df = pd.DataFrame({'A' : []})
term_doc = ref_vect.vectorize_tfidf(df)
X = term_doc.todense()
print(X)

featured_words = ref_vect.get_label_names()
print(featured_words)


#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Scale the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#build machine learning model
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), activation='logistic', 
                    solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

# mean accuracy on the given data and labels.
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

predicted = mlp.predict(X_test)
print("Prediction")
print(predicted)

#Result analysis
#set of labels predicted for a sample must exactly match the 
#corresponding set of labels in y_true
print("Accuracy of prediction")
print(accuracy_score(y_test, predicted))

precision, recall, fscore, support = all_score(y_test, predicted)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

data = [precision, recall, fscore, support]
row_labels = ['precision', 'recall', 'fscore', 'support']
col_labels = np.array(list(target_labels.keys()))
print(col_labels)
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(row_labels)))

table = plt.table(cellText=data, rowLabels=row_labels, rowColours=colors, 
          colLabels=col_labels,loc='top', clip_box='None')
cur_axes = plt.gca()
#cur_axes.axes.get_xaxis().set_visible(False)
#cur_axes.axes.get_yaxis().set_visible(False)
cur_axes.axis('off')
plt.show()



#train_class_prob = np.column_stack((X_train,y_train))
#test_class_prob = np.column_stack((X_test, y_test))
#print("Training set log probability: %f" % mlp.predict_proba(np.asmatrix(y_train)))
#print("Test set log probability: %f" % mlp.predict_proba(np.asmatrix(y_test)))
#print(confusion_matrix(y_test.values.argmax(axis=1),predictions.argmax(axis=1)))
#print(classification_report(y_test,predictions))

