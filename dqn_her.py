import time
import random
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import concatenate, Input, Conv2D, Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
import memory_her
import keras.backend as K
from constants import *
class DeepQ:

    def __init__(self, outputs, memorySize, discountFactor, learningRate, img_rows, img_cols, img_channels ,useTargetNetwork):

        self.output_size = outputs
        self.memory = memory_her.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0

        self.board = TensorBoard(log_dir='./log/tf', \
                        histogram_freq=0, write_graph=False, write_images=True)
        if K.backend() == 'tensorflow':
            with KTF.tf.device(TF_DEVICE):
                config = tf.ConfigProto(allow_soft_placement = True)
                config.gpu_options.allow_growth = True
                KTF.set_session(tf.Session(config=config))
                self.initNetworks()
        else :
            self.initNetworks()

    def initNetworks(self):

        self.model = self.createModel()
        if self.useTargetNetwork:
            self.targetModel = self.createModel()

    def createModel(self):
        input_shape = (self.img_channels, self.img_rows, self.img_cols)
        if K.image_dim_ordering() == 'tf':
            input_shape = (self.img_rows, self.img_cols, self.img_channels)

        state = Sequential(name='state_conv')
        state.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape))
        state.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
        state.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
        state.add(Flatten())

        # Now let's get a tensor with the output of our state:
        state_input = Input(input_shape, name='state_input')
        encoded_state = state(state_input)

        goal = Sequential(name='goal_conv')
        goal.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape))
        goal.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
        goal.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
        goal.add(Flatten())

        # Now let's get a tensor with the output of our goal:
        goal_input = Input(input_shape, name='goal_input')
        encoded_goal = goal(goal_input)

        merged = concatenate([encoded_state, encoded_goal], \
                                name='state_goal_input')

        fc = Dense(512, activation='relu', name='fc')(merged)
        model_output = Dense(self.output_size, activation='softmax', \
                            name='action')(fc)

        model =  Model(inputs=[state_input, goal_input], outputs=model_output)

        model.compile(optimizer=Adam(lr=self.learningRate), loss='mean_squared_error', metrics=['mae','accuracy'])
        model.summary()


        return model


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)
        print ('update target network')

    # predict Q values for all the actions
    def getQValues(self, state):
        if self.useTargetNetwork:
            predicted = self.targetModel.predict(state)
        else:
            predicted = self.model.predict(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        was_rand = False
        if rand < explorationRate :
            was_rand = True
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        # action = self.getMaxIndex(qValues)
        return action, was_rand

    def addMemory(self, state, stateGoal, action, reward, newState, \
                    newStateGoal, isFinal):
        self.memory.addMemory(state, stateGoal, action, reward, newState, \
                                newStateGoal, isFinal)


    def getMemorySize(self):
        return self.memory.getCurrentSize()


    def learnOnMiniBatch(self, miniBatchSize,):

        self.count_steps += 1

        state_batch, goals_batch, action_batch,reward_batch,newState_batch, _,isFinal_batch, \
             = self.memory.getMiniBatch(miniBatchSize)

        qValues_batch = self.model.predict([np.array(state_batch), \
                        np.array(goals_batch)],batch_size=miniBatchSize)

        isFinal_batch = np.array(isFinal_batch) + 0

        if self.useTargetNetwork:
            qValuesNewState_batch = self.targetModel.predict_on_batch([np.array(newState_batch), \
                                               np.array(goals_batch)])
        else :
            qValuesNewState_batch = self.model.predict_on_batch([np.array(newState_batch), \
                                         np.array(goals_batch)])

        Y_sample_batch = reward_batch + (1 - isFinal_batch) * self.discountFactor * np.max(qValuesNewState_batch, axis=1)

        X_batch = [np.array(state_batch),np.array(goals_batch)]
        Y_batch = np.array(qValues_batch)

        for i,action in enumerate(action_batch):
            Y_batch[i][action] = Y_sample_batch[i]

        self.model.fit(X_batch, Y_batch, validation_split=0.0, batch_size = miniBatchSize, epochs=1, verbose = 0, callbacks=[self.board])

        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()


    def saveModel(self, path):
        if self.useTargetNetwork:
            self.targetModel.save(path)
        else:
            self.model.save(path)

    def loadWeights(self, path):
        self.model.load_weights(path)
        if self.useTargetNetwork:
            self.targetModel.load_weights(path)


    def feedforward(self,observation,explorationRate):
        qValues = self.getQValues(observation)
        action, was_rand = self.selectAction(qValues, explorationRate)
        return action, was_rand
