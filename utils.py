def add_channel(image):
    gray_image = image/255.0
    gray_image = gray_image.reshape((1,image.shape[0],image.shape[1],1))
    #Debug to confirm the image is passed as floats
    #cv2.imshow("original",image)
    #cv2.imshow("float",gray_image[0,:,:,0])
    #cv2.waitKey(0)
    #exit()
    return gray_image

def get_observation_images(obs):
    observation = []
    observation.append(add_channel(obs[0].image))
    observation.append(add_channel(obs[1]))
    return observation

def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'r-')
