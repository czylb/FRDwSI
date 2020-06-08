import numpy as np
from sklearn import preprocessing
data = np.load('test_output/runs/synth_arch-344-9_D-951_Z-9_Nd-24192_V-116_noise-0.100000_activation-None_outActivations-None_model-neural_run-0.npz')
lst = data.files
# To compute the topic distribution per document, sum the document-topic prior and the final topic samples
# print(lst)
docTopic = data['doc_representations']+data['final_samples']
docTopic = np.array(docTopic)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_minMax = min_max_scaler.fit_transform(docTopic)
print(X_minMax)
final = open('final.txt','w')
finaldata = []
for i in X_minMax:
	p = i
	index = np.where(i == sorted(p)[-1])
	print(index)
	finaldata.append(index[0][0])

print(set(finaldata))
print(finaldata)
final.write(str(finaldata))
final.close();
print(len(finaldata))