# CharLM

## 1. Trigram Language Model

### Overview
This is the first part of `CharLM`, trigram-based character-level language model designed to generate sample names and evaluate probabilities of sequences in a dataset. The model supports **two training methods**:
1. **Count-based Trigram Model**: A statistical model based on counts of character trigrams.
2. **Neural Network-based Trigram Model**: A deep learning model trained using one-hot encoding of character pairs and a linear transformation.

The model generates synthetic names based on learned probabilities and evaluates the dataset using **negative log likelihood**.


### Installation and Usage

#### Dependencies
The code requires the following libraries:
- Python 3.8+
- PyTorch
- argparse

Install PyTorch using:
```bash
pip install torch
```

#### How to Run
1. **Prepare a Text File**:
   - Create a text file where each line represents a word or name (e.g., `names.txt`).

2. **Run the Model**:
   Use the following command to train and evaluate the model:
   ```bash
   python charlm.py --file_path <path_to_file> --method <count/neural> --epochs <num_epochs> --lr <learning_rate>
   ```

   **Example**:
   ```bash
   python charlm.py --file_path names.txt --method neural --epochs 200 --lr 50
   ```

### Arguments
| Argument       | Description                                                                                  | Default       |
|----------------|----------------------------------------------------------------------------------------------|---------------|
| `--file_path`  | Path to the text file containing the dataset.                                                | Required      |
| `--method`     | Training method: `count` for count-based or `neural` for neural network-based training.       | `count`       |
| `--epochs`     | Number of epochs (only for neural network-based training).                                    | `200`         |
| `--lr`         | Learning rate (only for neural network-based training).                                      | `50`          |

---

### Model Details

#### 1. Count-Based Trigram Model
- **How It Works**:
  - Counts the frequency of each trigram (sequence of three characters).
  - Calculates probabilities with **smoothing** to handle unseen trigrams.
- **Loss Function**:
  - Negative log likelihood of the dataset.

#### 2. Neural Network-Based Trigram Model
- **How It Works**:
  - Uses one-hot encoding of character pairs as input.
  - Learns a weight matrix (`W`) through gradient descent.
- **Loss Function**:
  - Negative log likelihood with L2 regularization.

---

### Output
1. **Training, Validation, and Test Loss**:
   - Displays performance metrics for the model.

2. **Generated Names**:
   - Outputs a list of 10 sample names generated by the trained model.

**Example Output**:
```
--------------------------------
Count Based Training
Train Loss: 2.5634
Val Loss: 2.7104
Test Loss: 2.6851
Sample names generated by the model:
Albina
Bert.
Cora.
Delta.
...
```

---

## Contributions
Feel free to contribute by improving the loss functions, adding new features, or testing on additional datasets.