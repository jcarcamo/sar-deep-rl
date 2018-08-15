# Machine Learning approach to use Unmanned Autonomous Vehicles (UAVs) in Search and Rescue (SAR)

## Abstract

Unmanned Aerial Vehicles (UAVs) are becoming more prevalent, more capable, and less expensive every day. Advances in battery life and electronic sensors have spurred the development of diverse UAV applications outside their original military domain. For example, Search and Rescue (SAR) operations stand to benefit greatly from modern UAVs since even the simplest commercial models are equipped with high-resolution cameras and the ability to stream video to a computer or portable device.  As a result, autonomous unmanned systems (terrestrial, marine, and aerial) have begun to be employed for such typical SAR tasks as terrain mapping, task observation, and early supply delivery. However, these systems were developed before recent advances in artificial intelligence such as Google Deepmind's breakthrough with the Deep Q-Network (DQN) technology.  Therefore, most of them rely heavily on Greedy or Potential-based heuristics, without the ability to learn.  In this research, we investigate a possible approximation (called Partially Observable Markov Decision Processes) for enhancing the performance of autonomous UAVs in SAR by incorporating newly-developed Reinforcement Learning methods. The project utilizes open-source tools such as Microsoft's state-of-the-art UAV simulator AirSim, and Keras, a machine learning framework that can make use of Google's popular tensor library called TensorFlow. The main approach investigated in this research is the Deep Q-Network.

## Requirements

This project uses [Unreal Engine 4](https://www.unrealengine.com/en-US/what-is-unreal-engine-4), and [Microsoft's AirSim](https://github.com/Microsoft/AirSim) as the vehicle simulator, Python 3.6 was used to build the environment and agent. [Keras](https://keras.io/) with [TensorFlow](https://www.tensorflow.org/) as Backend was used for the Agent implementation.

### Python dependencies

Python 3.6 was used for this development. All dependencies can be found in the ```requirements.txt``` file. Below is an example of the most visible ones.

- TensorFlow 1.10
- Keras 2.1.3
- opencv-python 3.4
- msgpack-rpc-python 0.4.1


### Installation

It is recommended to use [anaconda](https://www.continuum.io/downloads) to install and manage your python environment.

```bash
conda create -n <env> python=3.6
activate <env>
cd <repo>
pip install -r requirements.txt
```

There is a submodule of an example Unreal Engine project for convenience. If you wish to use it, issue the following commands:

```bash
cd sar-airsim-env
git submodule init
git submodule update
```

After that please follow the [AirSim Instructions](https://github.com/Microsoft/AirSim/blob/297f1c49d3ef0a1a0f0d841ef7dafa89603db327/docs/unreal_custenv.md) to properly setup the AirSim plugin into the Unreal Engine project.

## Acknowledgments

This implementation is inspired in the following repositories [AirGym](https://github.com/Kjell-K/AirGym) and [gym-unrealcv](https://github.com/Kjell-K/AirGym). Both of these repositories contain exceptional examples of what was needed to get started with [Unreal Engine 4](https://www.unrealengine.com/en-US/what-is-unreal-engine-4), and [Microsoft's AirSim](https://github.com/Microsoft/AirSim).
