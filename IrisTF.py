
# coding: utf-8

# # IRIS dataset investigation with Tensorflow

# ## Prasad Gharpure

# The dataset will be loaded from a sqlite database that has 150 entries of the iris dataset.  I will attempt to train a neural network to recognize the species from the provide petal and sepal dimensions.

# In[1]:

import sqlite3 as sqlite
import tensorflow as tf
import numpy as np
import sys
import math

# In[2]:

class DataFeeder:
    def __init__(self, size, theDataArray, theLabels):
        self.size = size
        self.theArray = theDataArray
        self.theLabels = theLabels
        self.reset()
        
    def getdata(self):
        return self.theArray[self.start:self.end]
    
    def getlabels(self):
        return self.theLabels[self.start:self.end]
    
    def moveforward(self):
        self.start = min(self.start+self.size, len(self.theArray))
        self.end = min(self.end+self.size, len(self.theArray))
        
    def isdone(self):
        return (self.start==self.end)
    def reset(self):
        self.start = 0
        self.end = min(self.size, len(self.theArray))
        
def make_dataset(nparray, hotlen):
    """take the dataset and return a dictionary with id, data, labels(onehot)"""
    ids = np.int32(nparray[0:, 0:1].flatten())
    data = nparray[0:, 1:5]
    labels = np.int32(nparray[0:,5:6].flatten())
    onehotlabels = np.zeros((len(labels), hotlen), dtype=np.int32)
    for i in range(0, len(labels)):
        onehotlabels[i, labels[i]] = 1
    return {
        "id" : ids,
        "data" : data,
        "onehot": onehotlabels
    }
    
def load_data():
    """Returns a dictionary with the training set, training labels, validation set and labels, and label names"""
    db = "database.sqlite"
    conn = sqlite.connect(db)
#    cursor = conn.execute("pragma table_info('Iris')")
#    for r in cursor:
#        print(r)
    sql = "select id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species from Iris"
    cursor = conn.execute(sql)
    rawdata = []
    labeldict = {}
    labelnames = []
    labelstart = 0
    for r in cursor:
        labelname = r[5]
        if not labelname in labeldict:
            labeldict[labelname] = len(labelnames)
            labelnames.append(labelname)
        labelvalue = labeldict[labelname]
        rawdata.append([r[0], r[1], r[2], r[3], r[4], labelvalue])
    npdata = np.array(rawdata)
    npdata_t = npdata.transpose()
    #normalization
    for i in range(1, 5):
        max = npdata_t[i].max()
        npdata_t[i] = npdata_t[i] * 1.0/max
    np.random.shuffle(npdata)
    split = int(len(npdata)*0.8)
    print("Splitting data at %s"%(split))
    train = npdata[0:split]
    test = npdata[split:]
    rv = {
        "train" : make_dataset(train, len(labelnames)),
        "test" : make_dataset(test, len(labelnames)),
        "labelnames": labelnames
    }
    return rv


class NetworkBuilder:
    def __init__(self):
        self.input_layer = None
        self.layers = None
        self.vars = []
        self.expected_output = None

    def build_network(self, layer_sizes):
        self.input_layer = tf.placeholder(tf.float32, (None, layer_sizes[0]), name="input")
        self.layers = [self.input_layer]
        for i in range(1, len(layer_sizes)):
            print('Building weights for layer %s'%( i))
            prev = layer_sizes[i-1]
            cur = layer_sizes[i]
            layer = {
                'weights': tf.Variable(tf.random_normal([prev, cur]), dtype=tf.float32, name="Wts_%s"%(i)),
                'biases' : tf.Variable(tf.random_normal([cur]), dtype=tf.float32, name="Bias_%s"%(i)),
                'output' : None,
            }
            self.vars.append(layer['weights'])
            self.vars.append(layer['biases'])
            self.layers.append(layer)

        for i in range(1, len(self.layers)):
            print('Connecting layer %s to %s'%(i-1, i))
            prev = self.layers[i-1]
            cur = self.layers[i]
            if type(prev) is dict:
                theinput = prev['output']
            else:
                theinput = prev
            cur['output'] = tf.nn.relu(tf.add(tf.matmul(theinput, cur['weights']), cur['biases']))
        # now do the output variable
        self.expected_output = tf.placeholder(tf.float32, (None, layer_sizes[-1]), name="labels")

    def make_feeder(self,inputdata, labeldata):
        return {self.input_layer: inputdata, self.expected_output: labeldata}

    def get_final_output(self):
        return self.layers[-1]['output']


# In[ ]:

#input_layer = tf.placeholder(tf.float32, (None, input_size), name="input")

#layers = [input_layer]
#vars_to_save = []
# for i in range(1, len(layer_sizes)):
#     print('Building weights for layer %s'%( i))
#     prev = layer_sizes[i-1]
#     cur = layer_sizes[i]
#     layer = {
#         'weights': tf.Variable(tf.random_normal([prev, cur]), dtype=tf.float32),
#         'biases' : tf.Variable(tf.random_normal([cur], dtype=tf.float32)),
#         'output' : None,
#     }
#     layers.append(layer)

# for i in range(1, len(layers)):
#     print('Connecting layer %s to %s'%(i-1, i))
#     prev = layers[i-1]
#     cur = layers[i]
#     if type(prev) is dict:
#         theinput = prev['output']
#     else:
#         theinput = prev
#     cur['output'] = tf.nn.relu(tf.add(tf.matmul(theinput, cur['weights']), cur['biases']))

# for i in range(1, len(layers)):
#     vars_to_save.append(layers[i]['weights'])
#     vars_to_save.append(layers[i]['biases'])
#layer1 = {
#    'weights' : tf.Variable(tf.random_normal([input_size, layer1_size]), dtype=tf.float32),
#    'biases' : tf.Variable(tf.random_normal([layer1_size]), dtype=tf.float32)
#}

# output1 = tf.nn.relu(tf.matmul(input, layer1['weights']) + layer1['biases'])
# layer2 = {
#     'weights' : tf.Variable(tf.random_normal([layer1_size, layer2_size]), dtype=tf.float32),
#     'biases' : tf.Variable(tf.random_normal([layer2_size]), dtype=tf.float32)
# }

# output2 = tf.nn.relu(tf.matmul(output1, layer2['weights']) + layer2['biases'])

# layer3 = {
#     'weights' : tf.Variable(tf.random_normal([layer2_size, output_size]), dtype=tf.float32),
#     'biases' : tf.Variable(tf.random_normal([output_size]), dtype=tf.float32)
# }

# final_output = tf.nn.relu(tf.matmul(output2, layer3['weights']) + layer3['biases'])
#final_output = layers[-1]['output']
# In[ ]:

class IrisTraining:
    def __init__(self, maxsteps, batchsize, errorLimit):
        self.maxsteps = maxsteps
        self.batch_size = batchsize
        self.error_limit = errorLimit
        self.datadict = load_data()
        self.input_size = len(self.datadict['train']['data'][0])
        self.output_size = len(self.datadict['labelnames'])
        self.layer_sizes = [self.input_size, 32, 16, 8, self.output_size]
        self.train_df = DataFeeder(self.batch_size, self.datadict['train']['data'], self.datadict['train']['onehot'])
        self.verify_df = DataFeeder(self.batch_size, self.datadict['test']['data'], self.datadict['test']['onehot'])
        self.nb = NetworkBuilder()
        self.nb.build_network(self.layer_sizes)
        self.saver = tf.train.Saver(self.nb.vars)
        self.final_output = self.nb.get_final_output()
        self.session = tf.Session()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.final_output, labels=self.nb.expected_output))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        init = tf.global_variables_initializer()
        self.session.run(init)

        
    def train(self):
        self.train_df.reset()
        for tstep in range(1, self.maxsteps):
            epoch_loss = 0
            while(not self.train_df.isdone()):
                feed = self.nb.make_feeder(self.train_df.getdata(), self.train_df.getlabels())
                _,err = self.session.run([self.optimizer, self.loss], feed)
                epoch_loss+=err
                self.train_df.moveforward()
            if tstep%10==0:
                print("Epoch %s Error %s"%(tstep, epoch_loss))
            self.train_df.reset()
            if epoch_loss < self.error_limit:
                print("Epoch %s Error %s"%(tstep, epoch_loss))
                break
        if epoch_loss < self.error_limit:
            self.saver.save(self.session, './mysession.save')
        return epoch_loss

    def verify(self):
        passcount = 0
        failcount = 0
        weight = np.array([4, 2, 1])
        self.verify_df.reset()
        while not self.verify_df.isdone():
            labels = self.verify_df.getlabels()
            feed = self.nb.make_feeder(self.verify_df.getdata(), self.verify_df.getlabels())
            res = self.session.run(tf.nn.softmax(self.final_output), feed)
            for i in range(0, len(labels)):
                idx = np.argmax(res[i])
                if labels[i][idx]==1:
                    print("PASS ", res[i], labels[i])
                    passcount = passcount+1
                else:
                    print("FAIL ", res[i], labels[i])
                    failcount = failcount+1
            self.verify_df.moveforward()
        print("%s %% PASS"%(int(10000.0*passcount/(passcount+failcount)/100)))
                    

def main():
    iris = IrisTraining(10000, 10, 0.3)
    if len(sys.argv) > 1:
        resname = sys.argv[1]
        print("FILENAME " + resname)
        iris.saver.restore(iris.session, resname)
        iris.verify()
    else:
        print("TRAINING")
        iris.train()
    
#     datadict = load_data()
#     trainsteps = 10000
#     batch_size = 5
#     input_size = 4
#     layer1_size = 50
#     layer2_size = 20
#     error_limit = 2
#     df = DataFeeder(batch_size, datadict['train']['data'], datadict['train']['onehot'])
#     layer_sizes = [input_size, 8, 16, 6, output_size]
#     print("Building network with %s layers"%(len(layer_sizes)))
#     nb = NetworkBuilder()
#     nb.build_network(layer_sizes)
#     final_output = nb.get_final_output()
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=nb.expected_output))
#     optimizer = tf.train.AdamOptimizer().minimize(loss)
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver(nb.vars)
#     sess = tf.Session()
#     sess.run(init)
# for tstep in range(1, trainsteps):
#     epoch_loss = 0
#     while(not df.isdone()):
#         feed = nb.make_feeder(df.getdata(), df.getlabels())
#         _,err = sess.run([optimizer, loss], feed)
#         epoch_loss+=err
#         df.moveforward()
#     if tstep%10==0:
#         print("Epoch %s Error %s"%(tstep, epoch_loss))
#     df.reset()
#     if epoch_loss < error_limit:
#         print("Epoch %s Error %s"%(tstep, epoch_loss))
#         break

# # In[ ]:
# if (epoch_loss < error_limit):
#     saver.save(sess, './mysession', global_step = tstep)
    

print(__name__)
if __name__ == "__main__":
    main()
