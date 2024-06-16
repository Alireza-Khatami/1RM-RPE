import pandas as pd
import numpy as np
import datetime

# Define the function to load the weights and biases
def load_model_weights(gender):
    weights = np.loadtxt(f'final_weights_{gender}.txt')
    biases = np.loadtxt(f'final_biases_{gender}.txt')
    return weights, biases

# Define the function to normalize the data
def normalize_data(df, scaler_params):
    scaled_df = pd.DataFrame()
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # Ensure the column is numeric
            mean = scaler_params[column]['mean']
            std = scaler_params[column]['std']
            scaled_df[column] = (df[column] - mean) / std
        else:
            scaled_df[column] = df[column]  # Keep non-numeric columns unchanged
    return scaled_df.values

# Load the data
data = pd.read_csv('data.csv')

# Load scaler parameters (these would have been saved during training)
x_scaler_params = {
    '1RM%': {'mean': 0.0, 'std': 1.0},  # Replace with actual mean and std
    'RPE': {'mean': 0.0, 'std': 1.0}    # Replace with actual mean and std
}
y_scaler_params = {'mean': 0.0, 'std': 1.0}  # Replace with actual mean and std

# Normalize the input data
X = normalize_data(data[['1RM%', 'RPE']], x_scaler_params)

# Load the model weights and biases
gender = 'both'  # Specify the gender if needed
weights, biases = load_model_weights(gender)

# Perform the forward pass using NumPy
def forward_pass(X, weights, biases):
    # ReLU activation function
    def relu(x):
        return np.maximum(0, x)
    
    # Layer 1
    z1 = np.dot(X, weights[:128].T) + biases[:128]
    a1 = relu(z1)
    # Layer 2
    z2 = np.dot(a1, weights[128:384].T) + biases[128:384]
    a2 = relu(z2)
    # Layer 3
    z3 = np.dot(a2, weights[384:512].T) + biases[384:512]
    a3 = relu(z3)
    # Layer 4
    z4 = np.dot(a3, weights[512:576].T) + biases[512:576]
    a4 = relu(z4)
    # Layer 5
    z5 = np.dot(a4, weights[576:].T) + biases[576:]
    a5 = relu(z5)
    return a5

# Get the predictions
predictions = forward_pass(X, weights, biases)

# Denormalize the predictions
predictions = predictions * y_scaler_params['std'] + y_scaler_params['mean']

# Save the results to a new CSV file
output_file = f'output_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'
output_df = pd.DataFrame(predictions, columns=['Predicted 1RM'])
output_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
