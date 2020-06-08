'''
Validate deep DMR implementation on synthetic data.

Adrian Benton
'''

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
sns.set(style="whitegrid")
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

import argparse, os, pickle, time
from copy import deepcopy
import numpy as np
import scipy.sparse
import theano
import theano.tensor as T

import mlp, opt
from neural_prior import buildSpriteParams, NeuralPrior
from lda_neural_prior import NeuralLDA

import pandas as pd

synth_output_dir = '../test_output/'

dataPathFmt  = os.path.join(synth_output_dir, 'data/synth_arch-%s_D-%s_Z-%d_Nd-%d_V-%d_noise-%f_activation-%s_outActivation-%s_run-%d.npz')
modelPathFmt = os.path.join(synth_output_dir, 'runs/synth_arch-%s_D-%s_Z-%d_Nd-%d_V-%d_noise-%f_activation-%s_outActivations-%s_model-%s_run-%d.npz')

def actToStr(fn):
  if fn is None:
    return 'None'
  else:
    try:
      return fn.name
    except:
      return fn.__name__

def genDataFromNetwork(architecture, D, Z, Nd, V, noise, activation=None, outActivation=None, learningRate=1.0, run=0):
  '''
  Generate corpus and observed supervision from a particular network architecture,
  sampling one-hot supervision for each document prior.
  '''
  
  print('Generating corpus from dDMR model...')
  print('Num Documents: %d\nNum Topics: %d\nNum Words per Document: %d\nVocabulary Size: %d\nSupervision Dimensionality: %d\nNeural Prior Architecture: %s\nActivation Function: %s\nStdDev of Supervision Noise: %.3f' % (D, Z, Nd, V, architecture[0], architecture[1:], actToStr(activation), noise))
  
  outWidth = architecture[-1]
  
  latentVals = np.random.randint(0, outWidth, D) # which prior to use for each document 
  trueSupervision = scipy.sparse.csr_matrix(([1.0 for v in latentVals],
                                             ([i for i, v in enumerate(latentVals)],
                                              latentVals))).todense().astype('float32')
  
  netWeights = []
  inWidth  = architecture[0]
  observedSupervision = theano.shared(0.01*np.random.randn(D, inWidth).astype('float32'),
                                      'input')
  net = observedSupervision
  allLayers = []
  
  for idx, (i, j) in enumerate(zip(architecture, architecture[1:])):
    
    netWeights.append(np.random.randn(j, i).astype('float32'))
    
    if idx >= len(architecture) - 2:
      if outActivation:
        net = outActivation(net.dot(netWeights[-1].T))
      else:
        net = net.dot(netWeights[-1].T)
    else:
      if activation:
        net = activation(net.dot(netWeights[-1].T))
      else:
        net = net.dot(netWeights[-1].T)
    allLayers.append(net)
  y = T.matrix('y')
  
  if outActivation == T.nnet.softmax:
    xent = T.nnet.categorical_crossentropy(T.clip(net, 1.e-12, 1.0 - 1.e-12), y)
    crossent_cost = xent.sum()
    cost = crossent_cost + 1.e-4 * (observedSupervision**2.).sum()/D # don't want input values to explode wildly
  else:
    cost = (T.sum((y - net)**2.0) + 1.e-4 * (observedSupervision**2.).sum())/D # MSE
  
  inGrad = T.grad(cost, [observedSupervision])
  
  optimizer = opt.AdadeltaOptimizer(learningRate=-learningRate, rho=0.95)
  
  updates = optimizer.getUpdates([observedSupervision], [inGrad[0]])
  
  updateObserved = theano.function(inputs=[y],
                                   outputs=net,
                                   updates=updates,
                                   allow_input_downcast=True)
  getCost = theano.function(inputs=[y], outputs=cost,
                                   allow_input_downcast=True)
  getActivations = theano.function(inputs=[], outputs=net,
                                   allow_input_downcast=True)
  
  for i in range(200):
    activations = updateObserved(trueSupervision)
    if not i % 10:
      print('Gen Data Iter %d: Cost %f' % (i, getCost(trueSupervision)))
  
  obsSup = observedSupervision.get_value() + noise*np.random.randn(D, inWidth)
  deltaVecs = np.exp(5.0 * np.random.choice(2, (outWidth, Z), p=[0.8, 0.2]) - 2.5) # dirichlet prior for topic preference for each label
  phiDists  = [np.random.dirichlet([0.1 for v in range(V)]) for z in range(Z)] # words preferred by each topic
  
  annotations = {'descriptor':obsSup.astype(np.float32),
                 'architecture':architecture,
                 'true_labels':activations,
                 'net_wts':netWeights,
                 'activation':activation,
                 'outActivation':outActivation,
                 'phi':phiDists,
                 'delta':deltaVecs,
                 'Z':Z,
                 'V':V,
                 'D':D,
                 'Nd':Nd}
  annNames = ['descriptor']
  # annDicts = {'descriptor':{i:'features_%d' % (i) for i in range(inWidth)}}
  annDicts = {'descriptor':{0: [0, 0, 0, 0, 0, 0, 1, 4, 0, 1, 0, 0, 0, 1], 1: [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 2: [0, 3, 0, 3, 0, 0, 5, 7, 0, 1, 0, 0, 0, 2], 3: [0, 1, 0, 3, 0, 1, 0, 3, 0, 1, 0, 0, 0, 0], 4: [0, 1, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], 5: [0, 2, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 6: [0, 7, 0, 8, 1, 0, 2, 3, 0, 1, 0, 0, 0, 0], 7: [0, 2, 0, 0, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0], 8: [0, 2, 0, 2, 0, 0, 1, 3, 0, 1, 0, 1, 1, 1], 9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], 10: [0, 1, 0, 4, 0, 4, 5, 20, 0, 0, 0, 1, 0, 0], 11: [1, 12, 0, 13, 3, 6, 6, 20, 1, 4, 0, 6, 0, 11], 12: [0, 0, 0, 0, 0, 0, 3, 6, 0, 1, 0, 0, 0, 0], 13: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 14: [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0], 15: [0, 3, 0, 5, 2, 0, 2, 8, 0, 0, 0, 0, 0, 1], 16: [0, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 17: [0, 1, 0, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1], 18: [0, 10, 0, 9, 1, 0, 0, 10, 0, 1, 0, 0, 0, 1], 19: [0, 6, 0, 9, 6, 3, 1, 19, 1, 2, 0, 2, 0, 1], 20: [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 0, 1, 0, 0], 21: [0, 3, 0, 5, 0, 0, 1, 13, 0, 0, 0, 0, 0, 0], 22: [3, 4, 0, 4, 0, 2, 1, 14, 0, 0, 0, 0, 1, 3], 23: [0, 1, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0], 24: [0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0], 25: [0, 1, 0, 0, 0, 0, 1, 7, 0, 0, 0, 1, 0, 6], 26: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2], 27: [0, 2, 0, 1, 1, 0, 0, 4, 0, 1, 0, 0, 0, 0], 28: [0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2], 29: [0, 1, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 2], 30: [0, 1, 0, 1, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1], 31: [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 32: [0, 2, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 2], 33: [0, 0, 0, 2, 0, 0, 0, 13, 0, 0, 0, 0, 4, 1], 34: [0, 4, 0, 3, 1, 0, 0, 9, 0, 0, 0, 0, 0, 0], 35: [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1], 36: [0, 5, 0, 8, 2, 0, 0, 4, 0, 1, 0, 0, 0, 5], 37: [1, 0, 0, 1, 0, 1, 0, 3, 0, 1, 0, 0, 0, 0], 38: [0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 1], 39: [0, 2, 0, 0, 0, 1, 2, 5, 1, 1, 0, 0, 0, 0], 40: [0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0], 41: [0, 1, 1, 2, 0, 0, 1, 7, 0, 1, 0, 0, 0, 1], 42: [0, 1, 0, 3, 0, 3, 1, 9, 0, 1, 0, 0, 0, 1], 43: [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 44: [0, 1, 0, 1, 0, 1, 0, 3, 0, 1, 0, 0, 0, 0], 45: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 46: [0, 2, 0, 4, 0, 1, 1, 4, 1, 0, 0, 1, 0, 1], 47: [0, 1, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0], 48: [0, 2, 0, 2, 0, 2, 0, 7, 0, 0, 0, 0, 0, 1], 49: [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 50: [0, 0, 0, 6, 0, 0, 0, 9, 0, 2, 0, 0, 0, 0], 51: [0, 2, 0, 5, 0, 0, 1, 13, 0, 0, 0, 0, 0, 0], 52: [0, 3, 0, 3, 2, 0, 0, 6, 0, 0, 0, 0, 0, 0], 53: [1, 2, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 54: [0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1], 55: [0, 2, 1, 5, 0, 1, 2, 11, 1, 3, 0, 1, 0, 2], 56: [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1], 57: [0, 2, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 1], 58: [0, 2, 0, 2, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0], 59: [0, 5, 0, 13, 3, 0, 0, 15, 0, 1, 0, 0, 0, 0], 60: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2], 61: [0, 3, 0, 2, 0, 1, 1, 5, 0, 0, 0, 0, 0, 0], 62: [0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0], 63: [0, 2, 0, 4, 1, 2, 3, 8, 1, 0, 0, 0, 0, 1], 64: [0, 8, 0, 10, 1, 4, 1, 11, 0, 2, 0, 0, 0, 3], 65: [0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0], 66: [0, 8, 0, 2, 0, 2, 3, 7, 0, 2, 0, 1, 0, 2], 67: [0, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1], 68: [0, 2, 0, 2, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0], 69: [0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 3], 70: [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3], 71: [0, 2, 0, 1, 0, 1, 0, 7, 0, 0, 0, 0, 0, 0], 72: [0, 0, 0, 2, 1, 1, 0, 3, 0, 0, 0, 0, 0, 1], 73: [0, 3, 0, 1, 0, 1, 0, 5, 0, 1, 0, 0, 0, 1], 74: [0, 2, 0, 0, 2, 0, 0, 6, 0, 0, 0, 0, 0, 1], 75: [0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0], 76: [0, 4, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 77: [0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 0, 0, 0, 0], 78: [0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0], 79: [0, 6, 0, 4, 1, 0, 2, 13, 0, 1, 0, 0, 0, 2], 80: [0, 0, 0, 10, 1, 3, 2, 10, 0, 1, 0, 0, 0, 1], 81: [0, 2, 0, 4, 2, 2, 0, 3, 0, 3, 0, 1, 0, 0], 82: [2, 25, 1, 22, 5, 16, 9, 45, 3, 6, 0, 5, 2, 11], 83: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 84: [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 2], 85: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2], 86: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 87: [0, 4, 0, 2, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], 88: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 89: [0, 4, 0, 1, 1, 0, 2, 12, 0, 1, 0, 0, 0, 0], 90: [0, 1, 0, 3, 0, 1, 0, 13, 0, 0, 0, 0, 0, 4], 91: [0, 2, 0, 2, 1, 2, 1, 5, 0, 0, 0, 0, 0, 0], 92: [0, 5, 0, 0, 1, 0, 1, 10, 0, 1, 0, 1, 0, 1], 93: [0, 2, 0, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 1], 94: [0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1], 95: [0, 1, 0, 8, 2, 2, 2, 13, 0, 0, 0, 0, 0, 2], 96: [0, 12, 1, 16, 3, 16, 2, 38, 1, 3, 0, 1, 0, 6], 97: [0, 5, 0, 2, 1, 2, 3, 5, 0, 2, 0, 1, 1, 1], 98: [0, 2, 0, 2, 0, 2, 0, 7, 0, 1, 0, 0, 0, 1], 99: [0, 3, 0, 5, 2, 0, 0, 14, 0, 0, 0, 0, 0, 0], 100: [0, 0, 0, 0, 0, 1, 0, 3, 0, 2, 0, 0, 0, 3], 101: [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0], 102: [0, 2, 0, 3, 1, 2, 0, 7, 0, 0, 0, 0, 0, 3], 103: [0, 6, 0, 9, 2, 1, 0, 6, 0, 1, 0, 0, 0, 0], 104: [2, 8, 0, 21, 4, 10, 2, 54, 0, 1, 0, 0, 0, 4], 105: [0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 106: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], 107: [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0], 108: [0, 0, 0, 2, 0, 0, 3, 6, 0, 1, 0, 0, 0, 1], 109: [0, 7, 0, 8, 2, 1, 1, 19, 1, 2, 0, 0, 1, 1], 110: [0, 2, 1, 6, 1, 2, 1, 8, 0, 3, 0, 0, 0, 2], 111: [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 112: [0, 3, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1], 113: [0, 2, 0, 2, 2, 3, 2, 10, 0, 3, 0, 0, 0, 1], 114: [0, 2, 0, 1, 1, 0, 0, 3, 0, 0, 0, 0, 0, 2], 115: [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0], 116: [0, 5, 0, 3, 0, 2, 1, 11, 0, 0, 0, 0, 0, 2], 117: [0, 20, 0, 32, 9, 13, 8, 66, 0, 4, 0, 1, 0, 5], 118: [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0], 119: [0, 3, 0, 2, 2, 1, 2, 10, 0, 1, 0, 0, 0, 2], 120: [0, 3, 0, 2, 0, 1, 0, 8, 0, 1, 0, 0, 0, 3], 121: [0, 0, 0, 0, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0], 122: [0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 123: [0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1], 124: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 125: [0, 7, 0, 7, 2, 0, 2, 11, 0, 0, 0, 0, 1, 4], 126: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 127: [0, 1, 0, 2, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0], 128: [0, 0, 0, 2, 0, 5, 1, 5, 0, 1, 0, 0, 0, 3], 129: [0, 2, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1], 130: [1, 1, 0, 4, 0, 0, 0, 10, 0, 0, 0, 0, 0, 1], 131: [0, 4, 0, 5, 1, 4, 6, 12, 1, 2, 0, 1, 1, 2], 132: [0, 1, 0, 1, 0, 5, 0, 3, 0, 0, 0, 0, 0, 0], 133: [0, 2, 0, 1, 1, 0, 1, 4, 0, 1, 0, 0, 0, 0], 134: [0, 4, 0, 7, 2, 4, 2, 28, 0, 2, 0, 0, 0, 2], 135: [0, 1, 0, 1, 0, 1, 0, 5, 0, 0, 0, 0, 0, 1], 136: [1, 4, 0, 17, 0, 6, 0, 28, 0, 7, 0, 0, 1, 6], 137: [0, 6, 0, 1, 2, 0, 0, 8, 0, 1, 0, 0, 0, 2], 138: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 139: [0, 2, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2], 140: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 141: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 142: [0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 143: [0, 6, 0, 12, 4, 0, 1, 18, 0, 1, 0, 0, 0, 4], 144: [0, 2, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0], 145: [0, 3, 0, 3, 0, 1, 1, 6, 0, 1, 0, 0, 0, 1], 146: [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], 147: [0, 4, 0, 7, 3, 3, 5, 21, 0, 4, 0, 1, 1, 5], 148: [0, 6, 0, 4, 2, 0, 0, 5, 0, 1, 0, 1, 0, 1], 149: [0, 8, 0, 6, 3, 1, 0, 9, 0, 3, 0, 1, 1, 4], 150: [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], 151: [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], 152: [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0], 153: [0, 2, 0, 2, 0, 2, 2, 2, 0, 1, 0, 1, 0, 2], 154: [0, 6, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 1], 155: [0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 156: [0, 0, 0, 1, 0, 1, 0, 9, 0, 0, 0, 0, 0, 0], 157: [0, 3, 0, 1, 0, 1, 0, 3, 0, 1, 0, 0, 0, 0], 158: [0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 0], 159: [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1], 160: [0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1], 161: [1, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 162: [0, 2, 2, 12, 3, 2, 2, 22, 0, 2, 0, 0, 3, 4], 163: [0, 4, 0, 4, 4, 0, 0, 9, 0, 0, 0, 0, 0, 0], 164: [0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1], 165: [0, 7, 0, 14, 1, 0, 1, 22, 0, 0, 0, 0, 0, 1], 166: [0, 1, 0, 2, 1, 0, 2, 3, 0, 0, 0, 0, 1, 2], 167: [0, 1, 0, 1, 0, 0, 1, 3, 0, 1, 0, 0, 1, 0], 168: [0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0], 169: [0, 4, 0, 1, 0, 4, 2, 13, 0, 0, 0, 2, 1, 1], 170: [0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0], 171: [0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0], 172: [0, 3, 0, 4, 0, 3, 1, 17, 0, 3, 0, 1, 0, 3], 173: [0, 1, 0, 3, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0], 174: [0, 0, 1, 6, 2, 0, 3, 5, 0, 1, 0, 0, 0, 3], 175: [0, 2, 0, 6, 1, 3, 1, 6, 0, 0, 0, 2, 0, 1], 176: [0, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 1], 177: [1, 23, 0, 27, 7, 10, 14, 54, 0, 5, 0, 3, 3, 11], 178: [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 179: [0, 4, 0, 3, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0], 180: [0, 1, 0, 2, 1, 2, 1, 6, 0, 3, 0, 0, 0, 0], 181: [0, 0, 0, 0, 1, 2, 0, 6, 0, 0, 0, 0, 0, 1], 182: [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1], 183: [0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0], 184: [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 185: [0, 1, 0, 2, 0, 1, 2, 4, 0, 2, 0, 0, 0, 0], 186: [0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 187: [0, 2, 0, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0], 188: [1, 21, 0, 15, 7, 6, 7, 44, 0, 6, 0, 5, 3, 5], 189: [0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0], 190: [0, 2, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 191: [0, 3, 0, 2, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0], 192: [0, 2, 0, 6, 0, 1, 0, 9, 0, 0, 0, 0, 0, 0], 193: [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0], 194: [0, 2, 0, 5, 0, 1, 0, 17, 0, 0, 0, 1, 0, 0], 195: [1, 2, 0, 1, 0, 2, 1, 2, 0, 2, 0, 1, 0, 0], 196: [0, 2, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0], 197: [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 198: [0, 5, 0, 5, 0, 1, 1, 5, 0, 0, 0, 2, 0, 1], 199: [0, 3, 0, 2, 0, 0, 1, 5, 0, 2, 0, 0, 0, 1], 200: [0, 5, 0, 3, 1, 0, 0, 13, 1, 0, 0, 0, 0, 1], 201: [0, 3, 0, 6, 0, 2, 1, 9, 0, 0, 0, 1, 1, 3], 202: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2], 203: [0, 0, 0, 0, 0, 4, 0, 3, 0, 0, 0, 0, 0, 0], 204: [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], 205: [0, 2, 0, 0, 0, 0, 1, 8, 0, 0, 0, 0, 0, 0], 206: [0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0], 207: [0, 4, 0, 3, 2, 2, 1, 6, 0, 2, 0, 1, 2, 4], 208: [0, 3, 0, 7, 1, 4, 0, 13, 0, 1, 0, 0, 0, 0], 209: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 210: [0, 1, 0, 3, 0, 1, 0, 7, 0, 1, 0, 0, 0, 0], 211: [0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 212: [0, 6, 0, 4, 1, 2, 1, 11, 0, 1, 0, 0, 1, 2], 213: [0, 3, 0, 4, 3, 3, 2, 13, 1, 1, 0, 0, 0, 2], 214: [0, 1, 0, 1, 0, 0, 2, 5, 0, 1, 0, 0, 0, 1], 215: [0, 4, 0, 3, 3, 3, 0, 12, 2, 3, 0, 0, 0, 1], 216: [0, 0, 0, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0], 217: [0, 0, 0, 2, 0, 2, 1, 4, 0, 2, 0, 0, 2, 2], 218: [0, 2, 0, 5, 1, 4, 2, 11, 0, 3, 0, 1, 0, 0], 219: [0, 2, 0, 1, 0, 1, 0, 11, 0, 0, 1, 0, 0, 0], 220: [0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 0, 0, 1], 221: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 222: [0, 0, 0, 2, 1, 0, 0, 12, 0, 0, 0, 0, 0, 0], 223: [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 224: [0, 1, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1], 225: [0, 3, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], 226: [0, 3, 0, 1, 0, 0, 0, 8, 0, 2, 0, 0, 2, 0], 227: [0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2], 228: [1, 5, 0, 4, 1, 2, 3, 9, 0, 2, 0, 1, 0, 2], 229: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 230: [0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 1, 0, 0], 231: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0], 232: [0, 2, 0, 10, 0, 1, 2, 15, 1, 3, 0, 1, 0, 0], 233: [0, 4, 0, 5, 1, 4, 1, 7, 1, 2, 0, 1, 0, 1], 234: [0, 0, 0, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0], 235: [0, 5, 0, 2, 1, 1, 1, 5, 0, 1, 0, 0, 0, 2], 236: [0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 237: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0], 238: [0, 2, 0, 5, 1, 1, 1, 21, 0, 0, 0, 0, 0, 1], 239: [0, 3, 0, 2, 0, 0, 0, 10, 0, 2, 0, 0, 0, 0], 240: [0, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], 241: [0, 1, 0, 3, 0, 1, 2, 7, 0, 0, 0, 0, 0, 1], 242: [0, 1, 0, 2, 0, 1, 1, 6, 0, 1, 0, 1, 2, 1], 243: [0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2], 244: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 245: [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 246: [0, 3, 0, 4, 1, 1, 1, 12, 0, 1, 0, 1, 0, 2], 247: [0, 0, 0, 1, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0], 248: [0, 4, 0, 2, 0, 2, 2, 8, 0, 0, 0, 0, 0, 2], 249: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2], 250: [0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 251: [0, 4, 1, 9, 5, 5, 4, 25, 0, 1, 0, 5, 0, 3], 252: [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], 253: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 254: [0, 2, 0, 1, 2, 0, 1, 3, 0, 1, 0, 0, 0, 2], 255: [0, 1, 0, 2, 1, 0, 0, 8, 0, 2, 0, 1, 0, 0], 256: [0, 3, 0, 2, 2, 0, 0, 5, 0, 1, 0, 0, 0, 0], 257: [0, 4, 0, 4, 0, 2, 3, 4, 0, 2, 0, 2, 0, 4], 258: [0, 4, 0, 8, 2, 2, 0, 9, 0, 0, 0, 1, 0, 2], 259: [0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 260: [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2], 261: [0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 2], 262: [0, 2, 0, 4, 1, 1, 0, 15, 0, 1, 0, 1, 0, 3], 263: [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], 264: [0, 0, 0, 0, 0, 2, 0, 5, 1, 0, 0, 0, 0, 0], 265: [0, 3, 0, 5, 2, 0, 0, 8, 0, 2, 0, 0, 0, 0], 266: [0, 2, 0, 1, 1, 0, 2, 13, 0, 0, 0, 0, 0, 0], 267: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 268: [0, 3, 1, 6, 1, 5, 2, 11, 0, 1, 0, 0, 0, 1], 269: [0, 5, 0, 8, 3, 0, 0, 9, 0, 3, 0, 0, 0, 0], 270: [0, 3, 0, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], 271: [1, 0, 0, 12, 2, 1, 1, 28, 0, 1, 0, 0, 3, 3], 272: [0, 6, 0, 4, 2, 4, 3, 15, 0, 0, 0, 0, 0, 2], 273: [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 274: [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 275: [0, 0, 0, 3, 0, 0, 2, 11, 0, 0, 0, 0, 0, 0], 276: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 277: [0, 2, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1], 278: [0, 1, 0, 3, 1, 0, 1, 5, 0, 1, 0, 0, 0, 0], 279: [0, 1, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0], 280: [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], 281: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 282: [0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1], 283: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 284: [0, 14, 0, 13, 9, 1, 1, 24, 1, 1, 0, 0, 0, 5], 285: [0, 0, 0, 1, 3, 4, 1, 8, 0, 0, 0, 0, 0, 1], 286: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 287: [0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 1], 288: [0, 2, 0, 2, 1, 0, 1, 4, 0, 0, 0, 0, 0, 0], 289: [0, 1, 0, 0, 0, 2, 0, 3, 0, 2, 0, 0, 0, 0], 290: [0, 1, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 291: [0, 3, 0, 10, 3, 0, 1, 12, 1, 2, 0, 0, 0, 2], 292: [0, 2, 0, 3, 0, 1, 0, 9, 0, 1, 0, 0, 0, 1], 293: [0, 5, 0, 6, 0, 11, 0, 17, 0, 0, 0, 0, 0, 1], 294: [0, 2, 0, 3, 1, 0, 1, 4, 0, 0, 0, 0, 0, 0], 295: [0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 296: [0, 0, 0, 1, 1, 0, 1, 11, 0, 1, 0, 0, 0, 1], 297: [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 298: [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0], 299: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 300: [0, 1, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0], 301: [0, 2, 0, 2, 1, 0, 0, 4, 0, 2, 0, 0, 0, 4], 302: [0, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 3], 303: [0, 2, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], 304: [0, 3, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 305: [0, 3, 0, 4, 0, 1, 0, 15, 0, 1, 0, 0, 0, 1], 306: [0, 2, 0, 7, 1, 5, 0, 15, 1, 0, 0, 0, 0, 7], 307: [0, 0, 0, 3, 1, 0, 3, 5, 0, 0, 0, 1, 9, 1], 308: [0, 0, 0, 3, 2, 0, 1, 4, 0, 0, 0, 2, 0, 0], 309: [0, 1, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0], 310: [0, 3, 0, 2, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1], 311: [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 312: [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 313: [0, 1, 0, 2, 0, 0, 1, 7, 0, 0, 0, 0, 0, 1], 314: [0, 1, 0, 1, 0, 1, 1, 9, 0, 1, 0, 1, 0, 3], 315: [0, 5, 0, 9, 1, 1, 0, 0, 0, 1, 0, 0, 0, 2], 316: [0, 5, 0, 4, 0, 0, 1, 8, 0, 1, 0, 0, 0, 3], 317: [0, 2, 0, 3, 0, 0, 0, 14, 0, 0, 0, 0, 0, 2], 318: [0, 4, 0, 3, 1, 0, 0, 6, 0, 2, 0, 0, 0, 0], 319: [0, 7, 0, 2, 0, 0, 0, 9, 1, 0, 0, 0, 0, 3], 320: [0, 0, 0, 1, 0, 2, 3, 10, 0, 0, 0, 0, 0, 0], 321: [0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0], 322: [0, 12, 0, 9, 2, 5, 4, 33, 0, 4, 0, 2, 2, 3], 323: [0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 3], 324: [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 2], 325: [0, 3, 0, 1, 0, 3, 0, 4, 0, 0, 0, 0, 0, 2], 326: [0, 3, 0, 4, 1, 1, 1, 5, 0, 1, 0, 1, 0, 0], 327: [0, 4, 0, 4, 2, 0, 1, 13, 0, 2, 0, 0, 0, 2], 328: [0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0], 329: [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2], 330: [0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0], 331: [1, 1, 0, 2, 0, 1, 2, 6, 0, 0, 0, 0, 0, 0], 332: [0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0], 333: [0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0], 334: [0, 2, 0, 3, 0, 1, 0, 10, 0, 2, 0, 0, 0, 2], 335: [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 336: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 337: [0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1], 338: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 339: [0, 3, 0, 7, 1, 1, 1, 15, 0, 1, 0, 0, 0, 0], 340: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], 341: [0, 4, 0, 4, 0, 2, 1, 11, 0, 0, 0, 0, 1, 2], 342: [0, 8, 0, 5, 0, 1, 0, 3, 1, 2, 0, 2, 0, 2], 343: [0, 2, 0, 8, 0, 5, 1, 19, 0, 1, 0, 0, 0, 1], 344: [0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0]}}
  tokenDict = {0: '0',1: '1',2: '2',3: '3',4: '4',5: '5',6: '6',7: '7',8: '8',9: '9',10: '10',11: '11',12: '12',13: '13',14: '14',15: '15',16: '16',17: '17',18: '18',19: '19',20: '20',21: '21',22: '22',23: '23',24: '24',25: '25',26: '26',27: '27',28: '28',29: '29',30: '30',31: '31',32: '32',33: '33',34: '34'35: '35'36: '36'37: '37'38: '38'39: '39'40: '40'41: '41'42: '42'43: '43'44: '44'45: '45'46: '46'47: '47'48: '48'49: '49',50: '50',51: '51',52: '52',53: '53',54: '54',55: '55',56: '56',57: '57',58: '58',59: '59',60: '60',61: '61',62: '62',63: '63',64: '64',65: '65',66: '66',67: '67',68: '68',69: '69',70: '70',71: '71',72: '72',73: '73',74: '74',75: '75',76: '76',77: '77',78: '78',79: '85',80: '88',81: '91',82: '95',83: '99',84: '103',85: '106',86: '115',87: '117',88: '120',89: '142',90: '147',91: '151',92: '153',93: '158',94: '163',95: '184',96: '185',97: '192',98: '203',99: '208',100: '213',101: '217',102: '218',103: '219',104: '222',105: '224',106: '229',107: '234',108: '237',109: '255',110: '256',111: '296',112: '300',113: '314',114: '323',115: '339'}
  payload = {}
  payload['annotations']      = annotations
  payload['annotation_dicts'] = annDicts
  payload['annotation_names'] = annNames
  payload['token_dict']       = tokenDict
  
  # sample distributions, topics, and words for each document
  Ds = []
  Ws = []
  
  docPriors = activations.dot(deltaVecs) + 1.e-8
  for d in range(D):
    topicDist = np.random.dirichlet(docPriors[d,:])
    
    Ds.extend([d for i in range(Nd)])
    
    # how many times each topic was sampled
    # Zs = np.random.choice(Z, Nd, replace=True, p=topicDist)
    
    # sample each word
    # for z in Zs:
    #   phi = phiDists[z]
    #   Ws.append(np.random.choice(V, p=phi))
  wsdata = open("../ws.txt",'r+')
  wsread = wsdata.read()
  wsread = wsread.strip("''[]")
  wss = wsread.split(',')
  for item in wss :
    Ws.append(int(item))
  wsdata.close()

  payload['Ds_body'] = np.asarray(Ds, dtype=np.int32)
  payload['Ws_body'] = np.asarray(Ws, dtype=np.int32)
  
  actName    = actToStr(activation)
  outActName = actToStr(outActivation)
  
  dataPath = dataPathFmt % ('-'.join([str(width) for width in architecture]), D, Z, Nd, V, noise, actName, outActName, run)
  
  np.savez_compressed(dataPath, **payload)

def test(inPath, outPath, architecture, activation, outActivation, onlyBias, annName='descriptor', initOracleWeights=False):
  d = np.load(inPath)
  
  annotations = d['annotations'].item()[annName].astype(np.float32)
  annotationDict = d['annotation_dicts'].item()[annName]
  tokenDict = d['token_dict'].item()
  
  print('Loaded data')
  
  Ds, Ws = d['Ds_body'], d['Ws_body']
  D = Ds.max()+1
  Z = d['annotations'].item()['Z']
  W = Ws.max()+1
  
  # Initalize network parameters
  L2 = 1.0
  
  betaMean, betaL1, betaL2 = 0.0, 0.0, L2
  deltaMean, deltaL1, deltaL2 = 0.0, 0.0, L2
  omegaMean, omegaL1, omegaL2 = 0.0, 0.0, L2
  deltaBMean, deltaBL1, deltaBL2 = 0.0, 0.0, L2
  omegaBMean, omegaBL1, omegaBL2 = 0.0, 0.0, L2
  
  params = buildSpriteParams(architecture[-1], W, Z, betaMean, betaL1,
                             betaL2, deltaMean, deltaL1,
                             deltaL2, omegaMean, omegaL1, omegaL2, deltaBMean, deltaBL1,
                             deltaBL2, omegaBMean, omegaBL1, omegaBL2, tieBetaAndDelta=True)
  
  optimizer = opt.AdadeltaOptimizer(-0.25)
  
  alphaGraph = mlp.MLP(12345, architecture, activation, outActivation,
                       optimizer, 0.0, L2, 'alphaGraph')
  alphaWts = alphaGraph.getWeights()
  origWts = deepcopy(alphaWts)
  if initOracleWeights and (len(architecture) > 1):
    wts = d['annotations'].item()['net_wts']
    idealWts = alphaWts
    for i, w in enumerate(wts):
      idealWts[i*2] = w.T
    
    alphaGraph.setWeights(idealWts)
  
  prior = NeuralPrior(params, alphaGraph, optimizer=optimizer, onlyBias=onlyBias)
  
  model = NeuralLDA(Z, W, prior, heldoutIndex=2, seed=12345)
  
  model.setData(Ws, Ds, annotations)
  
  print('Built model')
  
  model.learn(iters=1000, burnin=100, numSamples=10, llFreq=50)
  
  # Print out topics
  model.printTopWords(tokenDict)
  
  payload = model.serialize(saveCorpus=True)
  
  payload['topics'] = model.getTopWords(tokenDict)
  payload['annotations'] = model.getTopAnnotations(annotationDict)
  
  omegaBTopic, supertopics = model.prior.getSupertopics(tokenDict, topn=20)
  payload['omegaBias_supertopic'] = omegaBTopic
  payload['omega_supertopics']    = supertopics
  payload['likelihood_history']   = model.ll_history
  
  np.savez_compressed(outPath, **payload)

def genSynthAndTest(architecture, D, Z, Nd, V, noise, activation=None, outActivation=None, annName='descriptor', run=0):
  learningRate = 1.0
  
  dataPath  = dataPathFmt  % ('-'.join([str(width) for width in architecture]), D, Z, Nd, V, noise, str(activation), str(outActivation), run)
  if not os.path.exists(dataPath):
    genDataFromNetwork(architecture, D, Z, Nd, V, noise, activation, outActivation, learningRate, run)
  else:
    print('Skipping building data...')
  
  # print('Fitting LDA')
  
  # actName = actToStr(activation)
  # outActName = actToStr(outActivation)
  
  # modelType = 'lda'
  # modelPath = modelPathFmt % ('-'.join([str(width) for width in architecture]), D,
  #                             Z, Nd, V, noise, actName, outActName, modelType, run)
  # test(dataPath, modelPath, [architecture[0]], None, None, True, annName, False)
  
  # print('Fitting DMR')
  
  # actName = actToStr(activation)
  # outActName = actToStr(outActivation)
  
  # modelType = 'dmr'
  # modelPath = modelPathFmt % ('-'.join([str(width) for width in architecture]), D,
  #                             Z, Nd, V, noise, actName, outActName, modelType, run)
  # test(dataPath, modelPath, [architecture[0]], None, None, False, annName, False)
  
  print('Fitting dDMR')
  
  actName = actToStr(activation)
  outActName = actToStr(None)
  
  modelType = 'neural'
  modelPath = modelPathFmt % ('-'.join([str(width) for width in architecture]), D,
                              Z, Nd, V, noise, actName, outActName, modelType, run)
  test(dataPath, modelPath, architecture, activation, None, False, annName, False)

def plotSynthResults(runDir='../test_output/runs/', plotDir='../test_output/plots/'):
  ''' Collect and print performance for different models. '''
  
  if not os.path.exists(plotDir):
    os.mkdir(plotDir)
  
  def getCorpusParams(path):
    flds = path.replace('.npz', '').split('_')[1:]
    return tuple(['-'.join(f.split('-')[1:]) for f in flds])
  
  modelPaths = [p for p in os.listdir(runDir) if p.endswith('.npz') and p.find('_run-') > -1]
  trainPPLs   = {}
  heldoutPPLs = {}
  iterations  = {}
  
  allParams = []
  
  for p in modelPaths:
    try:
      model = np.load(os.path.join(runDir, p))
    except Exception as ex:
      import pdb; pdb.set_trace()
    
    params = getCorpusParams(p)
    actFn = params[6] if len(params) > 7 else 'sigmoid'
    key   = (int(params[0].split('-')[0]), float(params[5]), actFn, int(params[-1])) # dimensionality, noise, activation fn
    modelName = params[-2]
    
    if modelName == 'lda':
      modelName = 'LDA'
    elif modelName == 'dmr':
      modelName = 'DMR'
    elif modelName == 'neural':
      modelName = 'dDMR'
    
    history = [o for i, o in enumerate(model['likelihood_history']) if (i % 5) == 0]
    
    if key not in trainPPLs:
      trainPPLs[key] = {};
      heldoutPPLs[key] = {};
      iterations[key] = {};
    
    trainPPLs[key][modelName]   = [o['train_ppl'] for o in history]
    heldoutPPLs[key][modelName] = [o['heldout_ppl'] for o in history]
    iterations[key][modelName]  = [o['iteration'].item() for o in history]
    
    allParams.append(params)
  
  MODELS = ['LDA', 'DMR', 'dDMR']
  
  # For each corpus, plot train/heldout perplexity curves
  for key in trainPPLs:
    if not (('LDA' in trainPPLs[key]) and ('DMR' in trainPPLs[key]) and ('dDMR' in trainPPLs[key]) and ('LDA' in heldoutPPLs[key]) and ('DMR' in heldoutPPLs[key]) and ('dDMR' in heldoutPPLs[key])):
      print('Skipping', key)
      continue
    
    try:
      pp = PdfPages(os.path.join(plotDir,
                                 'supdim-%d_noise-%.3f_activation-%s_run-%d.pdf' % key))
      fig, ax = plt.subplots()
      
      ax.plot(iterations[key]['LDA'], trainPPLs[key]['LDA'],   'r^:', label='LDA-train')
      ax.plot(iterations[key]['LDA'], heldoutPPLs[key]['LDA'], 'r^--', label='LDA-dev')
      ax.plot(iterations[key]['DMR'], trainPPLs[key]['DMR'],   'gv:', label='DMR-train')
      ax.plot(iterations[key]['DMR'], heldoutPPLs[key]['DMR'], 'gv--',
              label='DMR-dev')
      ax.plot(iterations[key]['dDMR'], trainPPLs[key]['dDMR'], 'b<:', label='dDMR-train')
      ax.plot(iterations[key]['dDMR'], heldoutPPLs[key]['dDMR'], 'b<--',
              label='dDMR-dev')
      ax.set_xlabel('Iteration', fontsize=24)
      ax.set_ylabel('Heldout Perplexity', fontsize=24)
      legend = ax.legend(loc='best', fontsize=18)
      
      plt.gcf().subplots_adjust(bottom=0.15)
      plt.axvline(x=100, color='black', linestyle='dashed')
      
      pp.savefig()
      pp.close()
    except Exception as ex:
      print('Exception:', ex)
      import pdb; pdb.set_trace()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='train and evaluate models on synthetic data')
  #parser.add_argument('--arch', required=True, metavar='LAYER_WIDTH',
  #                    nargs='+', type=int,
  #                    help='hidden layer widths in the alpha network architecture')
  parser.add_argument('--arch', metavar='LAYER_WIDTH',
                      nargs='+', type=int,
                      help='hidden layer widths in the alpha network architecture')
  #parser.add_argument('--noise', required=True, metavar='NOISE', type=float,
  #  help='standard deviation of gaussian noise to perturb supervision')
  parser.add_argument('--noise', default=0.1, metavar='NOISE', type=float,
    help='standard deviation of gaussian noise to perturb supervision')
  parser.add_argument('--D', type=int, default=951, help='number of documents to generate')
  parser.add_argument('--Z', type=int, default=9,
                      help='number of topics')
  parser.add_argument('--V', type=int, default=116,
                      help='vocabulary size')
  parser.add_argument('--Nd', type=int, default=24192,
                      help='number of tokens per document')
  parser.add_argument('--nonlinear', action='store_true',
                      help='use sigmoid activations in hidden layer')
  parser.add_argument('--run', type=int, default=0,
                      help='which run')
  args = parser.parse_args()
  
  args.arch = [344, 9] if not args.arch else args.arch
  
  if not args.nonlinear:
    act, outAct = None, None
    print('Only linear activation functions')
  else:
    act, outAct = T.nnet.sigmoid, None
    print('Using nonlinear activation functions')

  data_dir  = os.path.join(synth_output_dir, 'data')
  run_dir   = os.path.join(synth_output_dir, 'runs')
  plot_dir  = os.path.join(synth_output_dir, 'plots')
  
  if not os.path.exists(synth_output_dir):
    os.mkdir(synth_output_dir)
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  if not os.path.exists(run_dir):
    os.mkdir(run_dir)
  if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
  
  genSynthAndTest(architecture=args.arch, D=args.D, Z=args.Z, Nd=args.Nd,
                  V=args.V, noise=args.noise, activation=act, outActivation=outAct,
                  run=args.run)
  # plotSynthResults(run_dir, plot_dir)
