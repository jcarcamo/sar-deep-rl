ENV_NAME = 'AirSimEnv-v0'

CONTINUE = False #load a pre-trained model
RESTART_EP = 0 # the episode number of the pre-trained model

TRAIN = True # train the network
USE_TARGET_NETWORK = False # use the target network

RANDOM_WALK=False
TF_DEVICE = '/gpu:0'
MAX_EPOCHS = 100000 # max episode number
MEMORY_SIZE = 50000
LEARN_START_STEP = 32

BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # 1e6
GAMMA = 0.95
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
MAX_EXPLORE_STEPS = 1000
TEST_INTERVAL_EPOCHS = 100
SAVE_INTERVAL_EPOCHS = 50

LOG_NAME_SAVE = 'log'
MODEL_DIR = LOG_NAME_SAVE + '/model' # the path to save deep model
PARAM_DIR = LOG_NAME_SAVE + '/param' # the path to save the parameters
VIZ_DIR = LOG_NAME_SAVE + '/viz' # the path to save viz
DATA_FILE = LOG_NAME_SAVE + '/data_trained.csv' # the path to save trajectory

LOG_NAME_READ = 'log'
#the path to reload weights, monitor and params
weights_path = LOG_NAME_READ + '/model/dqn_her_ep' + str(RESTART_EP)+ '.h5'
params_json = LOG_NAME_READ + '/param/dqn_her_ep' + str(RESTART_EP) + '.json'
