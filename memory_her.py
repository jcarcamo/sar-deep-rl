import random
import numpy as np

class Memory:
    """
    Modified to include goals as part of the Experience Replay
    """

    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.statesGoals = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.newStatesGoals = []
        self.finals = []

    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index):
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index],'statesGoals': self.statesGoals[index], 'newStatesGoals': self.newStatesGoals[index]}

    def addMemory(self, state, stateGoal, action, reward, newState, \
                    newStateGoal, isFinal) :
        if (self.currentPosition >= self.size - 1) :
            self.currentPosition = 0
        if (len(self.states) > self.size) :
            self.states[self.currentPosition] = state
            self.statesGoals[self.currentPosition] = stateGoal
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.newStatesGoals[self.currentPosition] = newStateGoal
            self.finals[self.currentPosition] = isFinal

        else :
            self.states.append(state)
            self.statesGoals.append(stateGoal)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.newStatesGoals.append(newStateGoal)
            self.finals.append(isFinal)

        self.currentPosition += 1

    def getMiniBatch(self, size) :
        state_batch = []
        stateGoal_batch = []
        action_batch = []
        reward_batch = []
        newState_batch = []
        newStateGoal_batch = []
        isFinal_batch = []

        indices = random.sample(list(np.arange(len(self.states))), min(size,len(self.states)) )
        for index in indices:
            state_batch.append(self.states[index][0])
            stateGoal_batch.append(self.statesGoals[index][0])
            action_batch.append(self.actions[index])
            reward_batch.append(self.rewards[index])
            newState_batch.append(self.newStates[index][0])
            newStateGoal_batch.append(self.newStatesGoals[index][0])
            isFinal_batch.append(self.finals[index])


        return state_batch, stateGoal_batch, action_batch, reward_batch, newState_batch, stateGoal_batch, isFinal_batch
