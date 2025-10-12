# Python-Chess

[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)  
[![Python version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)  

A chess engine / AI project built in Python (with SQL backend, machine learning, and a frontend) aiming to emulate the author’s playing style via adaptive models.  

**Expected completion date**: October 10, 2025  

---

## Table of Contents

- [About](#about)  
- [Features](#features)  
- [Architecture / Tech Stack](#architecture--tech-stack)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [Configuration](#configuration)  
  - [Running](#running)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [Roadmap](#roadmap)  
- [License](#license)  
- [Contact](#contact)  

---

## About

This project is an attempt to build a chess AI that mimics a player’s style by training models on their game data. It combines chess logic, data storage, and machine learning to evolve an adaptive opponent.  

It is being developed using Python, SQL, and specialized chess / ML libraries. Future improvements include increasing the size of the dataset, along with incorporating a Reinforcement Learning with Human Feedback (RLHF) model. 

---

## Features

- Chess move validation, game state management  
- Adaptive AI opponents trained on personal game data  
- ML Analysis, combined usage of convolutional, recurrent, and graph neural networks. 
- Frontend (UI) + backend API  
- Persistence (SQL database) for storing games, model checkpoints, etc.  
- Logging, metrics, evaluation  

---

## Architecture & Tech Stack

| Component | Technology / Library |
|-----------|----------------------|
| Language | Python 
| Database | SQL | MySQL (or any local database instance) |
| Chess Logic | python-chess | chess.pgn |
| ML / AI | PyTorch / PyTorch Geometric |
| Frontend | Vue.js / HTML / TypeScript / etc. |
| API / Backend | Flask |

---

## Getting Started

### Prerequisites

- Python 3.8+  
- pip 
- A SQL database (e.g. MySQL for local dev)  
- GPU if training large models (Optional, i.e. "cuda" engine)
- node / npm for frontend functionality  

### Installation

```bash
git clone https://github.com/AbElo-77/Python-Chess.git
cd Python-Chess


python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows


pip install pytorch
pip install pytorch-geometric
pip install pandas
```

### Configuration 

Choose the NN model you want to train, Convolutional, Recurrent, or Graph.  

### Running

``` bash
python -m backend.index # Run The Flask Backend 

cd frontend
npm run dev 
```

## Usage

- Upload 10-20K personal games or use a public dataset (recommended).
- - Lichess for multi-million game PGN files, but can splice for the first couple thousand games. 
- Trained chosen model using the same class number, len(move_to_id). 

## Roadmap

Upcoming planned improvements include

1. Alpha release: basic AI vs human with learned model + shallow search
2. Analytics dashboard: graphs of your style, mistakes, openings repertoire
3. Tuning control: user can shift AI style weight (aggressive, passive, unpredictable)
4. Self-play mode: model plays against itself to generate synthetic training data
5. Multiplayer mode / online server support (WebSocket integration)
6. Persisted user accounts, game history sync
7. Support for variant chess modes (Chess960, bughouse)
8. Model improvements: transformer-based models, reinforcement learning, meta-learning

## Contributing 

Please contribute using PEP8 guidelines and updating the README, or contact me using the contacts below. 

## License 

This project is licensed under the MIT license. 

## Contacts

- GitHub: AbElo-77
- Email: abdallaelokely@gmail.com
