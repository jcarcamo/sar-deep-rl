#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import csv

import numpy as np
import random
import os

from environment import  *
from utils import *
from constants import *

from matplotlib import pyplot as plt
import dqn_her


if __name__ == '__main__':

    env = Env(shaped_reward=False)
    obs_new = env.reset()

    ACTION_SIZE = 4
    # #ACTION_LIST = env.discrete_actions
    INPUT_HEIGTH=obs_new[0].image.shape[0]
    INPUT_WIDTH=obs_new[0].image.shape[1]
    INPUT_CHANNELS = 1

    succeed = 0

    #HER
    K = 4
    #load init param
    if not CONTINUE:
        explorationRate = INITIAL_EPSILON
        current_epoch = 1
        stepCounter = 0
        loadsim_seconds = 0
        Agent = dqn_her.DeepQ(ACTION_SIZE, MEMORY_SIZE, GAMMA, LEARNING_RATE, \
                          INPUT_HEIGTH, INPUT_WIDTH, INPUT_CHANNELS, \
                          USE_TARGET_NETWORK)
    else:
        #Load weights and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            explorationRate = d.get('explorationRate')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')
            succeed = d.get('succeed')
            if succeed is None:
                succeed = 0
            Agent = dqn_her.DeepQ(ACTION_SIZE, MEMORY_SIZE, GAMMA, \
                              LEARNING_RATE, INPUT_HEIGTH, INPUT_WIDTH,  \
                              INPUT_CHANNELS, USE_TARGET_NETWORK)
            Agent.loadWeights(weights_path)

    #main loop
    try:
        start_time = time.time()
        epoch_data = []
        for epoch in range(current_epoch, MAX_EPOCHS, 1):
            positionsList = [[],[]]
            agent_decision = 0

            obs = env.reset()
            observation = get_observation_images(obs)
            positionsList[0].append(obs[0].position.x_val)
            positionsList[1].append(obs[0].position.y_val)
            cumulated_reward = 0
            if (epoch % TEST_INTERVAL_EPOCHS != 0 or stepCounter < LEARN_START_STEP) and TRAIN is True :  # explore
                EXPLORE = True
            else:
                EXPLORE = False
                print ("Evaluate Model")

            #borrowed from HER example
            episode_experience = []
            episode_succeeded = False

            for t in range(1000):

                start_req = time.time()

                if EXPLORE is True: #explore
                    s = observation[0]
                    g = observation[1]
                    action, was_rand = Agent.feedforward(observation, explorationRate)
                    if not was_rand:
                        agent_decision += 1
                    # if np.random.rand(1) < 0.5:
                    #     action = np.random.randint(ACTION_SIZE)
                    obs_new, reward, done = env.step(action)
                    newObservation =  get_observation_images(obs_new)

                    s_next = newObservation[0]
                    stepCounter += 1
                    # Agent.addMemory(observation[0], observation[1], action, \
                    #                 reward, newObservation[0], \
                    #                 newObservation[1],  done)
                    episode_experience.append((s,action,reward,s_next,g, done))
                    observation = newObservation
                    obs = obs_new
                #test
                else:
                    if not RANDOM_WALK:
                        action, _ = Agent.feedforward(observation,0)
                    else:
                        action, _ = Agent.feedforward(observation,1)
                    print(action)
                    obs_new, reward, done = env.step(action)
                    newObservation =  get_observation_images(obs_new)
                    observation = newObservation
                    obs = obs_new

                cumulated_reward += reward
                if reward == 0:
                    episode_succeeded = True
                    succeed += 1

                positionsList[0].append(obs[0].position.x_val)
                positionsList[1].append(obs[0].position.y_val)

                if done:
                    print("Agent taking decisions", agent_decision,"times out of",t+1,"steps.",100*agent_decision/float(t+1),"%")
                    m, sec = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                    h, m = divmod(m, 60)

                    if stepCounter == LEARN_START_STEP:
                        print("Starting learning")

                    #Episode is done, let's add some experience to memory
                    if TRAIN:
                        for replay in range(t):
                            s, a, r, s_n, g, d = episode_experience[t]
                            Agent.addMemory(s, g, a, r, s_n, g, d)
                            # K-future strategy
                            for k in range(K):
                                future = np.random.randint(replay, t)
                                _, _, _, g_n, _, d = episode_experience[future]
                                final = np.allclose(s_n, g_n)
                                #only working with sparsed reward
                                r_n = 0 if final else -1
                                Agent.addMemory(s, g_n, a, r_n, s_n, g_n, d)

                        if Agent.getMemorySize() >= LEARN_START_STEP:
                            Agent.learnOnMiniBatch(BATCH_SIZE)
                            #print("Episode Done, Learning time")
                            if explorationRate > FINAL_EPSILON and stepCounter > LEARN_START_STEP:
                                explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / MAX_EXPLORE_STEPS

                    print ("EP " + str(epoch) +" Csteps= " + str(stepCounter) + " - {} steps".format(t + 1) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(round(explorationRate, 2)) + "  Success rate=" + str(round(succeed/float(epoch)*100, 2)) + "  Time: %d:%02d:%02d" % (h, m, sec))
                    epoch_data.append([str(epoch), str(stepCounter),t + 1, str(
                        round(cumulated_reward, 2)), str(round(explorationRate, 2)), str(round(succeed/float(epoch)*100, 2)), "%d:%02d:%02d" % (h, m, sec)])
                    with open(DATA_FILE, 'a') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        for data in epoch_data:
                            wr.writerow(data)
                    epoch_data = []
                    # SAVE SIMULATION DATA
                    if epoch % SAVE_INTERVAL_EPOCHS == 0 and TRAIN is True:
                        # save model weights and monitoring data
                        print ('Save model')
                        Agent.saveModel(MODEL_DIR + '/dqn_her_ep' + str(epoch) + '.h5')

                        parameter_keys = ['explorationRate', 'current_epoch','stepCounter', 'FINAL_EPSILON','loadsim_seconds','succeed']
                        parameter_values = [explorationRate, epoch, stepCounter,FINAL_EPSILON, int(time.time() - start_time + loadsim_seconds), succeed]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(PARAM_DIR + '/dqn_her_ep' + str(epoch) + '.json','w') as outfile:
                            json.dump(parameter_dictionary, outfile)


                    plt.title('Traveled path of agent on epoch '+str(epoch))

                    #for i in np.arange(0,len(positionsList[0]),2):
                    #    if(i+2 <= len(positionsList[0])):
                    #        plt.plot(positionsList[0][i:i+2],positionsList[1][i:i+2],'k-')
                    #plt.savefig(VIZ_DIR + '/dqn_her_ep' + str(epoch) + '.png', format='png')
                    plt.scatter(positionsList[0],positionsList[1])
                    plt.xlim(0, 20)
                    plt.ylim(-10, 10)
                    plt.plot(positionsList[0],positionsList[1])
                    plt.savefig(VIZ_DIR + '/dqn_her_ep' + str(epoch) + '.png', format='png')
                    plt.close()
                    break

    except KeyboardInterrupt:
        print("Shutting down")
