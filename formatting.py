import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# edges = np.load("train_edges.npy")

# file = open("train.txt", "w")
# for i in range(edges.shape[0]):
#     file.write(str(edges[i][0] + 1) + " " + str(edges[i][1] + 1) + "\n")
# file.close()

f = open("PPI.1", "r") 
header = f.readline().split()
adj = np.zeros((3890, int(header[1])))
for line in f:
    emb = line.split()
    i = int(emb[0]) - 1
    adj[i] = np.array(map(float, emb[1:]))

f.close()

adj_rec = np.dot(adj, adj.T)

test_edges = np.load("test_edges.npy")
test_edges_false = np.load("test_edges_false.npy")
test_edges = tuple(zip(*test_edges))
test_edges_false = tuple(zip(*test_edges_false))

edges_pos = test_edges
edges_neg = test_edges_false

preds = sigmoid(adj_rec[edges_pos])
preds_neg = sigmoid(adj_rec[edges_neg])

preds_all = np.hstack([preds, preds_neg])
labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
roc_score = roc_auc_score(labels_all, preds_all)
ap_score = average_precision_score(labels_all, preds_all)

print((roc_score, ap_score))