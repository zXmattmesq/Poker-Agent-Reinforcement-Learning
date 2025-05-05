# Poker-Agent-Reinforcement-Learning

## Overview

This GitHub folder contains a Python-based implementation of a poker game with a Reinforcement Learning (RL) agent. It includes the game environment, AI agent models, training scripts, simulation tools, and a graphical user interface (GUI) to play against the AI.

## Key Components

* `Back_End/`: Contains the core game logic and AI components.
    * `envs.py`: Defines the poker environment.
    * `models.py`:  Contains the neural network model for the AI agent.
    * `utils.py`:  Provides utility functions (state encoding, etc.).
* `Front_End/`:  Implements the GUI.
    * `main_ui.py`:  The main script for launching the GUI.
* `Scripts/`:  Includes scripts for training and simulating the agent.
    * `train.py`:  Script to train the RL agent.
    * `simulate.py`:  Script to run simulations.
* `main.py`:  The main entry point to run training, simulation, or the GUI.
* `code_walkthrough.ipynb`:  A Jupyter Notebook providing a code walkthrough.

##  Prerequisites

* Python 3.x
* PyTorch
* Tkinter (usually included with Python)
* Other libraries (numpy, etc.) -  Install using `pip install -r requirements.txt` (if a requirements.txt file is present)

## GUI

You can run the GUI by opening code_walkthrough.ipynb and running the GUI command at the bottom of the file

##  Download

You can download this GitHub folder by using the "Clone or download" button on the GitHub repository page.  Choose "Download ZIP" and extract the contents. Alternatively, use git:

```bash
git clone <https://github.com/zXmattmesq/Poker-Agent-Reinforcement-Learning>
