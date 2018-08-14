# Machine Learning approach to use Unmanned Autonomous Vehicles (UAVs) in Search and Rescue (SAR)

## Abstract

Unmanned Aerial Vehicles (UAVs) are becoming more common every day. Advances in battery life and electronic sensors have made possible the development of many different applications outside the military domain. Search and Rescue (SAR) operations can greatly benefit from using these vehicles since even the simplest commercial models are equipped with high resolution cameras and the ability to stream video to a computer or portable device. Some common applications of UAVs in SAR are: mapping, victim search, task observation, and early supply delivery. Autonomous Unmanned Systems: ground, aquatic, and aerial; have been developed in the last decade, however these systems were developed before Google Deepmind breakthrough with Deep Q-Network (DQN), hence most of them heavily rely on greedy or potential-based heuristics, without the ability to learn. In this work we present some possible approximations to advance the research in SAR UAVs by incorporating Reinforcement Learning methods using open-source tools such as Microsoft's state-of-the-art simulator called AirSim and Google's Machine Learning framework TensorFlow. Two different approaches are proposed to be implemented: Deep Recurrent Q-Network (DRQN)..

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
