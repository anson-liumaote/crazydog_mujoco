
## Dependencies
- *Python* - 3.8
- [*PyTorch* - 1.10.0 with CUDA 11.3](https://pytorch.org/get-started/previous-versions/)
- *Crazydog project* (https://github.com/anson-liumaote/crazydog.git)

## Installation
1. Clone this repository
    ```bash
    git clone git@github.com:morrisx28/crazydog_mujoco.git
    ```
2. Create the conda environment:
    ```bash
    conda env create -f environment.yml
    ```

## Quick Start
1. Replay the RL policy on Crazydog (Biped wheel robot):
    ```bash
    conda activate rlmujoco
    cd crazydog_urdf
    python mujoco_rl_pos_6dof.py config/crazydog.yaml
    ```
    All supported two version of biped wheel robot.


