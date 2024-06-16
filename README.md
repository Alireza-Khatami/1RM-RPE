
# Re-Evaluating the Calculation of 1RM Based on 1RM% and RPE

This repository contains the code and data associated with the paper "Re-Evaluating the Calculation of 1RM Based on 1RM% and RPE." Our method employs a neural network to predict the one-repetition maximum (1RM) from input features such as 1RM% and Rating of Perceived Exertion (RPE).

## Overview

One-repetition maximum (1RM) is a key measure in strength training, representing the maximum weight that an individual can lift for one repetition. Traditional methods to estimate 1RM often rely on percentages of 1RM and RPE scores. This paper presents a novel approach using a Multi-Layer Perceptron (MLP) to improve the accuracy and generalizability of 1RM predictions.

## Repository Contents

- `train_model.py`: Script to train the MLP model on the provided dataset.
- `evaluate.py`: Script to evaluate the trained model on new data.
- `data.csv`: Template CSV file for entering new data for evaluation.
- `final_weights_both.txt`, `final_biases_both.txt`: Pre-trained model weights and biases for evaluation.
- `combined-data.csv`: Example dataset used for training.
- `README.md`: This document.

## Installation and Setup

To run the code, you'll need Python 3.8 and the following Python packages:
- NumPy
- Pandas

You can install the required packages using pip:
```sh
pip install numpy pandas
```

## Data Preparation

For evaluation, prepare your data in the `data.csv` file. The file should contain the following columns:
- `1RM%`
- `RPE`

Ensure the file is empty except for the headers before adding your data.

## How to Evaluate the Model

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/re-evaluating-1rm.git
   cd re-evaluating-1rm
   ```

2. **Prepare your data:**
   - Enter your data in the `data.csv` file located in the repository. Make sure the file is empty except for the headers.

3. **Run the evaluation:**
   ```sh
   python evaluate.py
   ```

4. **Check the results:**
   - The predictions will be saved in a new file named `output_{current-date}.csv` in the repository directory.

## Training the Model

The model was trained using a 5-fold cross-validation approach on a combined dataset of male and female participants. The training process involved:

1. **Data Normalization:**
   - Normalizing the input features (1RM% and RPE) to have zero mean and unit variance.

2. **Model Architecture:**
   - A Multi-Layer Perceptron (MLP) with the following architecture:
     - Input Layer: Size of input features (2: 1RM% and RPE)
     - Hidden Layer 1: 128 neurons, ReLU activation
     - Hidden Layer 2: 256 neurons, ReLU activation
     - Hidden Layer 3: 128 neurons, ReLU activation
     - Hidden Layer 4: 64 neurons, ReLU activation
     - Output Layer: 1 neuron (predicted 1RM)

3. **Training:**
   - Using Mean Squared Error (MSE) as the loss function.
   - Optimizing with Adam optimizer.
   - Training for 100 epochs per fold.

4. **Saving the Model:**
   - The final weights and biases of the model were saved to text files for use during evaluation.

## Citation

If you use this code or dataset in your research, please cite our paper:

```
@article{yourname2024reevaluating1rm,
  title={Re-Evaluating the Calculation of 1RM Based on 1RM% and RPE},
  author={Dr. Amin Fashi-Professor of Sports and Science at Shahid Beheshti University-,
 Alireza Khatami-master of multi-media at Shahid Behesti University-, 
Javid Moghadam- master of Sports and Science at Shahid Beheshti University-},
  journal={??},
  year={2024},

}
```


## Contact

For any questions or issues, please contact us at alirza.khatami.research@gmail.com or open an issue in this repository.

---

Feel free to adjust the details as necessary to fit your specific project and preferences.
