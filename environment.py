from airsimapi import  *

class State():
    def __init__(self, image, position, orientation):
        self.image = image
        self.position = position
        self.orientation = orientation

class Env():
    def __init__(self, shaped_reward = False):
        self.client = customAirSimClient()
        self.state = State(self.client.getDepthImage(), \
                            self.client.getPosition(), \
                            self.client.getOrientation())
        self.shaped_reward = shaped_reward
        self.reward_factor = 1
        self.reward_th =  0.05
        self.desired_goal = np.load("desired_goal.npy")

    def reward(self, boxes):
        reward = 0
        # if self.shaped_reward: # Not sparsed reward
        for box in boxes:
            # get reward of all boxes considering the size and position of box
            (xmin, ymin), (xmax, ymax) = box
            boxsize = (ymax - ymin) * (xmax - xmin)
            x_c = (xmax + xmin) / 2.0
            x_bias = x_c - 0.5
            discount = max(0, 1 - x_bias ** 2)
            reward += discount * boxsize

        if reward > self.reward_th:
            if self.shaped_reward: # Not sparsed reward
                reward = min(reward * self.reward_factor, 10)
            else:
                reward = 0
            print ('Get ideal Target!!!')
        elif reward == 0:
            reward = -1
            # print ('Get Nothing')
        else:
            if self.shaped_reward: # Not sparsed reward
                reward = 0
            else:
                reward = -1
            # print ('Get small Target!!!')
        # else: # Sparsed reward, to test HER
        #     for box in boxes:
        #         (xmin, ymin), (xmax, ymax) = box
        #         # print ("xmax:",(xmax),"xmin:", (xmin),"xmax-xmin:",(xmax-xmin))
        #         # print ("ymax:",(ymax),"ymin:", (ymin),"ymax-ymin:",(ymax-ymin))
        #         boxsize = (ymax - ymin) * (xmax - xmin)
        #         reward += boxsize
        #     print ("Reward:",reward)
        #     if reward > self.reward_th:
        #         reward = 0
        #         print ('Found object')
        #     else:
        #         reward = -1
        #         print ('Get Nothing')
        return reward

    def step(self, action):
        done = False
        reward = -1
        collided = self.client.move(action)
        observation = self.client.getDepthImage()
        self.state = State(observation, \
                            self.client.getPosition(), \
                            self.client.getOrientation())
        if not collided:
            #check if object is in image.
            object_mask = self.client.getSegementationImage()
            boxes = self.client.getBboxes(object_mask, ['Goal'])
            reward = self.reward(boxes)
            if self.shaped_reward:
                if reward > 0:
                    done = True
            else:
                if reward == 0:
                    done = True
        else:
            if self.shaped_reward:
                reward = -100
            else:
                reward = -1
            done = True
        return [self.state, self.desired_goal], reward, done

    def reset(self, size = None):
        self.client.reset()
        self.state = State(self.client.getDepthImage(), \
                            self.client.getPosition(), \
                            self.client.getOrientation())
        return [self.state,self.desired_goal]
