import numpy as np
import matplotlib.pyplot as plt
import random

def seq2image(seq,Q=10,N=40):
  img = np.zeros(400,int)
  b = [pow(2,i) for i in range(Q)]
  for i in range(N):
    btoken = format(seq[i], 'b')
    btoken = np.append([0]*(Q-len(btoken)),[btoken[j] for j in range(len(btoken))])
    btoken = btoken[::-1]
    img[i*Q:i*Q+Q] = btoken
  img = img.reshape(20,20)
  return img

a = np.load('./dataset/testing_sequences_ordered.npy')
b = np.load('./dataset/testing_labels_ordered.npy')
for i in range(10):
  idx = random.randint(0,len(b))
  img = seq2image(a[idx])
  plt.imshow(img)
  plt.title(b[idx])
  plt.xticks(np.arange(len(img)))
  plt.show()