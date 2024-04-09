import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Load test data
data_dict = pickle.load(open('./data.pickle', 'rb'))
x_test = np.asarray(data_dict['data'])
y_test = np.asarray(data_dict['labels'])

# Evaluate model
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
conf_matrix = confusion_matrix(y_test, y_predict)

# Print results
print('Performance Metrics:')
print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1-score: {:.2f}'.format(f1))

labels_dict = {0: 'Hello', 1: 'goodBye', 2: 'I love You', 3: 'Yes', 4:'No', 5:'thank you'}

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels_dict.values(), yticklabels=labels_dict.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Save evaluation results
with open('performance_analysis.txt', 'w') as f:
    f.write('Performance Metrics:\n')
    f.write('Accuracy: {:.2f}%\n'.format(accuracy * 100))
    f.write('Precision: {:.2f}\n'.format(precision))
    f.write('Recall: {:.2f}\n'.format(recall))
    f.write('F1-score: {:.2f}\n'.format(f1))
