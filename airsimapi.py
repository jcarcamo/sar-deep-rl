import time
import sys
import cv2
from PIL import Image

#Using AirSimClient from AirSim Repo
sys.path.append('C:/Users/juan_/git/AirSim/PythonClient')
from AirSimClient import *

class customAirSimClient(MultirotorClient):
    def __init__(self):
        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True)
        self.armDisarm(True)
        self.takeoff()

        # hack instead of a classifier for objects
        # TODO include the real classifier
        # success = self.simSetSegmentationObjectID("SkySphere", 42, True)
        # print('success', success)
        success = self.simSetSegmentationObjectID("[\w]*", 0, True)
        print('Setting everything to 0', success)
        # success = self.simSetSegmentationObjectID("birch[\w]*", 2, True)
        # print('success', success)
        # success = self.simSetSegmentationObjectID("fir[\w]*", 2, True)
        # print('success', success)
        # success = self.simSetSegmentationObjectID("hedge[\w]*", 5, True)
        # print('success', success)
        #success = self.simSetSegmentationObjectID("tree[\w]*", 255, True)
        success = self.simSetSegmentationObjectID("Goal", 2)
        print('Setting our goal', success)
        #wait until it takes off
        counter = 0
        while self.getPosition().z_val > -1.45:
            print("taking off: ", self.getPosition().z_val)
            time.sleep(0.2)
        self.z = self.getPosition().z_val


    def straight(self, duration, speed):
        pitch, roll, yaw  = self.getPitchRollYaw()
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.moveByVelocityZ(vx, vy, self.z, duration, DrivetrainType.ForwardOnly)
        start = time.time()
        return start, duration

    def yaw_right(self, duration):
        self.rotateByYawRate(30, duration)
        start = time.time()
        return start, duration

    def yaw_left(self, duration):
        self.rotateByYawRate(-30, duration)
        start = time.time()
        return start, duration
    """
    # According to AirSim docs, this are the drone available cameras
    # cam_id=0, center front
    # cam_id=1, left front
    # cam_id=2, right front
    # cam_id=4, center rear
    #  https://github.com/Microsoft/AirSim/blob/e169e65755f4254c29b67617f50502a262e64e63/docs/image_apis.md
    """
    def getSceneImage(self, cam_id):
        responses = self.simGetImages([ \
            ImageRequest(cam_id, AirSimImageType.Scene, False, False), \
            ])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 4
        img_rgba = img1d.reshape(response.height, response.width, 4)
        #discard alpha, to make image smaller
        #TODO check if it influence training
        img_rgb = img_rgba[:,:,:-1]
        #cv2.imshow("Scene", img_rgba)
        #cv2.waitKey(0)
        return img_rgb[:,:,::-1]

    def getDepthImage(self):
        rawImage = self.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True, False)])[0]
        # Taken from original AirGym, transforms from float img to int image
        img1d = np.array(rawImage.image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (rawImage.height, rawImage.width))
        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))

        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark
        depth = (maxIntensity)*(image/maxIntensity)**factor
        depth = np.array(depth, dtype=np.uint8)
        return depth

    def getSegementationImage(self):
        rawImage = self.simGetImages([ \
            ImageRequest(0, AirSimImageType.Segmentation, False, False)])[0]
        img1d = np.fromstring(rawImage.image_data_uint8, dtype=np.uint8) #get numpy array
        img_rgba = img1d.reshape(rawImage.height, rawImage.width, 4)
        img_rgb = img_rgba[:,:,:-1]
        # print(np.unique(img_rgba[:,:,0], return_counts=True)) #red
        # print(np.unique(img_rgba[:,:,1], return_counts=True)) #green
        # print(np.unique(img_rgba[:,:,2], return_counts=True)) #blue
        return img_rgb

    #Harcoded currently
    def getMask(self,object_mask):
        #[r,g,b] = 94, 253, 92
        #lower_range = np.array([g-50,b-50,r-25])
        #upper_range = np.array([g+2,b+50,r+25])
        [r,g,b] = 193, 103, 112
        lower_range = np.array([b-3,g-3,r-3], dtype='uint8')
        upper_range = np.array([b+3,g+3,r+3], dtype='uint8')
        mask = cv2.inRange(object_mask, lower_range, upper_range)
        return mask

    def getBbox(self,object_mask,object):
        #only get an object's bounding box
        width = object_mask.shape[1]
        height = object_mask.shape[0]
        mask = self.getMask(object_mask)
        output = cv2.bitwise_and(object_mask, object_mask, mask = mask)
        # cv2.imshow("mask",mask)
        # cv2.imshow("side-by-side",np.hstack([object_mask, output]))
        # cv2.waitKey(0)
        nparray = np.array([[[0, 0]]])
        pixelpointsCV2 = cv2.findNonZero(mask)

        if type(pixelpointsCV2) == type(nparray):# exist target in image
            x_min = pixelpointsCV2[:,:,0].min()
            x_max = pixelpointsCV2[:,:,0].max()
            y_min = pixelpointsCV2[:,:,1].min()
            y_max = pixelpointsCV2[:,:,1].max()
            #print x_min, x_max ,y_min, y_max
            box = ((x_min/float(width),y_min/float(height)),#left top
                   (x_max/float(width),y_max/float(height)))#right down
        else:
            box = ((0,0),(0,0))

        return mask , box

    def getBboxes(self, object_mask, targets):
        boxes = []
        for obj in targets:
            mask,box = self.getBbox(object_mask, obj)
            boxes.append(box)
        return boxes

    def move(self, action):
        start = time.time()
        duration = 0
        collided = False

        if action == 0:
            start, duration = self.straight(1, 4)
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    return True
            self.moveByVelocity(0, 0, 0, 1)
            self.rotateByYawRate(0, 1)


        if action == 1:
            start, duration = self.yaw_right(0.8)
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    return True
            self.moveByVelocity(0, 0, 0, 1)
            self.rotateByYawRate(0, 1)

        if action == 2:
            start, duration = self.yaw_left(1)
            while duration > time.time() - start:
                if self.getCollisionInfo().has_collided == True:
                    return True
            self.moveByVelocity(0, 0, 0, 1)
            self.rotateByYawRate(0, 1)

        if action == 3:
            fake = True
            #start, duration = self.straight(1, 4)
            #while duration > time.time() - start:
            #    if self.getCollisionInfo().has_collided == True:
            #        return True
            #self.moveByVelocity(0, 0, 0, 1)
            #self.rotateByYawRate(0, 1)

        return collided

    def reset(self):
        super().reset()
        time.sleep(0.2)
        self.enableApiControl(True)
        self.armDisarm(True)
        self.takeoff()
        #wait until it takes off
        while self.getPosition().z_val > -1.45:
            print("taking off: ", self.getPosition().z_val)
            time.sleep(0.2)
            counter +=1
            if counter == 5:
                this.reset()
        self.z = self.getPosition().z_val
        return self.getDepthImage()
