actual_file = open('testing/output.txt', 'r', encoding='utf8')
pred_file = open('testing/pred_output.txt', 'r', encoding='utf8')

actual, pred = [], []

for line in actual_file:
    actual.append(int(line[0]))

for line in pred_file:
    pred.append(int(line[0]))

matrix = [[0, 0], [0, 0]]
actual_yes, actual_no, predicted_yes = 0, 0, 0 

length = len(actual)
for i in range(length):
    if actual[i] == 1:
        actual_yes += 1
    elif actual[i] == 0:
        actual_no += 1
    if pred[i] == 1:
        predicted_yes += 1
    matrix[actual[i]][pred[i]] += 1

TP = matrix[1][1]
TN = matrix[0][0]
FP = matrix[0][1]
FN = matrix[1][0]

total = length
accuracy = (TP + TN) / total
misclassfication = (FP + FN) / total
recall = TP / actual_yes
specificity = TN / actual_no
precision = TP / predicted_yes
f_score = 2 * ((recall * precision) / (recall + precision))

print("Confusion Matrix:", matrix)
print("Accuracy: ", accuracy)
print("Misclassfication Rate: ", misclassfication)
print("True Positive Rate (Recall): ", recall)
print("True Negative Rate (Specificity): ", specificity)
print("Precision: ", precision)
print("F Score: ", f_score)