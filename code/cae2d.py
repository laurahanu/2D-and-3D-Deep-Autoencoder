#CONVOLUTIONAL AUTOENCODER Using Lasagne used on an MRI image dataset of shape 180,1,160,64,64 for nr of images, nr of channels, depth, image height, image width 

from lasagne.layers import InputLayer, NINLayer, flatten, reshape, Upscale3DLayer, DenseLayer
from lasagne.layers import get_output, get_output_shape, get_all_params, get_all_layers 
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import binary_crossentropy as bce
from lasagne.objectives import squared_error
from lasagne.updates import adam

import numpy as np
import theano
from theano import tensor as T
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 

from skimage.io import imsave
import scipy.misc
import scipy.io
floatX=theano.config.floatX

inPath = '' #path to dataset

outPath = '' #path to where you want the results to be saved

def get_args():
	print ('getting args...')

def save_args():
	print ('saving args...')

def build_net(nz=200):

	input_depth=160
	input_rows=64
	input_columns=64

	F=32 

	# encoder

	enc = InputLayer(shape=(None,1,64,64))
	enc = Conv2DLayer(incoming=enc, num_filters=F*2, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)
	enc = Conv2DLayer(incoming=enc, num_filters=F*4, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)
	enc = Conv2DLayer(incoming=enc, num_filters=F*8, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)
	enc = Conv2DLayer(incoming=enc, num_filters=F*8, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)
	enc = reshape(incoming=enc, shape=(-1,F*8*4*4))
	enc = DenseLayer(incoming=enc, num_units=nz, nonlinearity=sigmoid)
	
	# decoder
	
	dec = InputLayer(shape=(None,nz))
	dec = DenseLayer(incoming=dec, num_units=F*8*4*4)
	dec = reshape(incoming=dec, shape=(-1,F*8,4,4))
	dec = Deconv2DLayer(incoming=dec, num_filters=F*8, filter_size=4, stride=2, nonlinearity=relu, crop=1)
	dec = Deconv2DLayer(incoming=dec, num_filters=F*4, filter_size=4, stride=2, nonlinearity=relu, crop=1)
	dec = Deconv2DLayer(incoming=dec, num_filters=F*2, filter_size=4, stride=2, nonlinearity=relu, crop=1)
	dec = Deconv2DLayer(incoming=dec, num_filters=1, filter_size=4, stride=2, nonlinearity=sigmoid, crop=1)

	return enc, dec

# get the shape of the network

enc, dec = build_net()
for l in get_all_layers(enc):
	print (get_output_shape(l))
for l in get_all_layers(dec):
	print (get_output_shape(l))


def prep_train(alpha=0.0002, beta=0.5, nz=200):
	
	E,D=build_net(nz=nz)

	x=T.Tensor4('x')
	
	# x -> symbolic variable, input to the computational graph

	#Get outputs z=E(x), x_hat=D(z)
	
	encoding = get_output(E,x)
	
	decoding = get_output(D,encoding)

	#Get parameters of E and D
	
	params_e=get_all_params(E, trainable=True)
	
	params_d=get_all_params(D, trainable=True)
	
	params = params_e + params_d

	#Calculate cost and updates
	
	cost = T.mean(squared_error(x,decoding))
	
	grad=T.grad(cost,params)

	updates = adam(grad,params, learning_rate=alpha , beta1=beta)	

	train = theano.function(inputs=[x], outputs=cost, updates=updates)
	
	rec = theano.function(inputs=[x], outputs=decoding)

	test = theano.function(inputs=[x], outputs=cost)
	
	#theano.function returns an actual python function used to evaluate our real data

	return train ,test, rec, E, D


def train( trainData, testData, nz=100, alpha=0.00005, beta=0.5, batchSize=128, epoch=100):
	
	train, test, rec, E, D = prep_train(nz=nz, alpha=alpha)
	
	print (np.shape(trainData) )
	
	sn,sc,sx,sy=np.shape(trainData) 
	
	print (sn,sc,sx,sy)
	
	batches=int(np.floor(float(sn)/batchSize))
	
	#initialize arrays to store the cost functions

	trainCost_=[]
	
	testCost_=[]
	
	print ('batches=',batches)

	timer=time.time()
		
	print ('epoch  \t batch  \t  train  \t  cost  \t    test  \t   cost   \t  time (s)')
	
	for e in range(epoch):

		for b in range(batches):

			trainCost=train(trainData[b*batchSize:(b+1)*batchSize]) 
			
			testCost=test(testData[:10]) #test first 10 images
			
			print e , '\t', b ,'\t', trainCost ,'\t' ,testCost ,'\t' ,time.time()-timer
			
			timer=time.time()
			
			trainCost_.append(trainCost)
			
			testCost_.append(testCost)
			
			#save results every 10 interations

			if b%100==0:

				# create a montage to visualize how close the decoded images are to the input images
				# the chosen method here creates a montage of 2x3 images with 3 real images on top of 3 decoded ones, but other ways to visualize the results can be used
			
				x_test1=x_test[200:300:10,:,:,:]
				
				montageRow1 = np.hstack(x_test1[:10].reshape(-1,64,64))
				
				REC=rec(x_test1[:10])
				
				montageRow2 = np.hstack(REC[:10].reshape(-1,64,64))
				
				montage = np.vstack((montageRow1, montageRow2))
								
				scipy.io.savemat(outPath + 'montageREC'+str(e)+'.mat',(dict(montage=montage)))


			# save plot of the cost functions

			plt.plot(range(e*batches+b),trainCost_[:e*batches + b])

			plt.plot(range(e*batches+b),testCost_[:e*batches + b]) 

			plt.legend()

			plt.xlabel('Iterations')

			plt.ylabel('Cost Function')

			plt.savefig(outPath + 'cost_regular_{}.png'.format(e))

	return test, rec, E, D

def test(x, rec):
	
	return rec(x)

# LOAD DATA

data=np.load(inPath,mmap_mode='r').astype(floatX)

data=np.transpose(data,(0,2,1,3))

sn,sc,sx,sy=np.shape(data) 

# create training and testing datasets

x_train=data[20:,:,:,:]

x_test=data[:20,:,:,:]

test, rec, E, D =train(x_train, x_test) 

# Save example reconstructions at the end of the iterations

REC = rec(x_test[:10])

print (np.shape(REC), np.shape(x_test[:10]))

fig=plt.figure()

newDir=''

montageRow1 = np.hstack(x_test[:3].reshape(-1,64,64).swapaxes(1,3))

montageRow2 = np.hstack(REC[:3].reshape(-1,64,64).swapaxes(1,3))

montage = np.vstack((montageRow1, montageRow2))

print ('montage shape is ',montage.shape)

np.save(outPath + 'montageREC.npy',montage)

scipy.io.savemat(outPath + 'montageREC.mat',(dict(montage=montage))) 
