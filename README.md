# AI Chess Engine

A Python-based chess engine that combines a deep neural network with a parallel Monte Carlo Tree Search (MCTS). Inspired by AlphaZero and techniques used in Leela Chess Zero, the engine uses a policy/value network, batched neural-network inference, and multiprocess tree parallelization.

## 🧠 Neural Network Architecture

<p align="center">
  <img height="800" alt="Neural Network Architecture" src="https://github.com/user-attachments/assets/1755e2fc-b93e-4372-9773-ec928731d2f3"/>
</p>

</br>

## ⚡ Multiprocess MCTS
* **Tree Parallelization:** multiple worker processes search a shared tree stored in shared memory.
* **WU-UCT (Watch the Unobserved):** tracks leaves whose evaluations are in progress. These temporary counts are incorporated into selection so workers are less likely to repeatedly choose the same path before earlier evaluations finish.
* **qFPU (First Play Urgency):** assigns an estimated initial value to unvisited actions based on the parent node's expected value, helping balance exploration and exploitation.
* **GPU Batching:** collects leaf positions into batches before neural-network evaluation to improve GPU utilization.

</br>

## Results
The first plot shows top-3 accuracy of the policy network, while the second compares game progress against Stockfish, with the random model as a baseline.

<table>
  <tr>
    <th>Policy Network Top-3 Accuracy</th>
    <th>Games Against Stockfish</th>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ade526cd-29a3-46a7-aa73-78b52a339bca" width="450" alt="Top-3 Accuracy"></td>
    <td><img src="https://github.com/user-attachments/assets/29497041-3abc-48b5-bd03-1e2ea885572d" width="450" alt="Games Against Stockfish"></td>
  </tr>
</table>


</br>

## 💻 CLI commands

### General
* `start-server` – Starts the chess engine server, listening for client connections via TCP sockets.
* `print-status` – Displays the status of the currently loaded model and server

### Model & Data Management
* `convert-games <input_dir> <output_train_dir> <output_test_dir>` – Encoding of `.pgn` game files into the `.rdg` format.
* `load-model <model_path>` – Loads pre-trained neural network
* `train-model <dataset_dir>` – Starts the training

### Evaluation
* `compare-model-to-stockfish <num_games>` – Plays test matches against Stockfish to evaluate engine
* `compare-mcts <num_sim_1> <num_sim_2>` – Pits two different MCTS configurations against each other
* `show-loss <test_data_dir> <models_dir>` – Generates learning curve plots (Policy & Value Loss)
* `top-k-accuracy <test_data_dir>` – Calculates Top-K accuracy for the Policy head

</br>

## ⚙️ Configuration (`config.json`)

The project includes a `config.json` file for easy parameter adjustments without modifying the code.

* **Startup Automation (`startCommands`):** Defines CLI commands to execute automatically at application startup.
* **Architecture & Hyperparameters:** Configures the neural network architecture, MCTS parameters, and data-processing settings.

</br>

## 📡 TCP Socket API

The server evaluates a chess position using a history of up to three FEN states.

### Request Format (Client -> Server)
```json
{
  "command": "get_move",
  "boards": [
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
  ]
}
```

### Response Format (Server -> Client)
```json
{
  "move": "b8c6"
}
