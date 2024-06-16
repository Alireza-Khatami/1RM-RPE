import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Define the neural network model
class RPEModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RPEModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x

def standardize_columns(X):
    XNormed = (X - X.mean()) / (X.std())
    return XNormed, {"mean": X.mean(), "std": X.std()}

# Function to train the model and log errors
def train_and_evaluate(X, y, gender, percentage):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    fold_correlations = []
    log_data = []
    input_size = X.shape[1]
    output_size = 1
    X, x_params = standardize_columns(X)
    y, y_params = standardize_columns(y)
    log_data.append(f"X params scaler: {x_params}")
    log_data.append(f"Y params scaler: {y_params}")
    model = RPEModel(input_size, output_size)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        
        # Training
        for epoch in range(150):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = mean_squared_error(y_val.numpy(), val_outputs.numpy())
            fold_results.append(val_loss)

            # Calculate correlation for the fold
            val_corr = np.corrcoef(y_val.numpy().flatten(), val_outputs.numpy().flatten())[0, 1]
            fold_correlations.append(val_corr)

            # Plot correlation for the fold
            plt.scatter(y_val.numpy(), val_outputs.numpy(), label=f'Fold {fold+1}')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Fold {fold+1} Correlation: {val_corr:.2f}')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'correlation_plot_fold_{fold+1}_{gender}.png')
            plt.clf()

            log_data.append(f'Fold {fold+1} - Validation Loss: {val_loss}')
            log_data.append(f'Fold {fold+1} - Correlation: {val_corr}')

    # Plot all fold correlations
    plt.plot(range(1, len(fold_correlations) + 1), fold_correlations, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Correlation')
    plt.title(f'Fold Correlations for {gender}')
    plt.grid(True)
    plt.savefig(f'fold_correlations_{gender}.png')
    plt.clf()

    avg_val_loss = np.mean(fold_results)
    log_data.append(f'Average Validation Loss for {gender}: {avg_val_loss}')
    log_data.append(f'Fold Correlations: {fold_correlations}')

    # Compute the final matrix product of the network's weights
    with torch.no_grad():
        fc1_weight = model.fc1.weight.data
        fc1_bias = model.fc1.bias.data
        fc2_weight = model.fc2.weight.data
        fc2_bias = model.fc2.bias.data
        fc3_weight = model.fc3.weight.data
        fc3_bias = model.fc3.bias.data
        fc4_weight = model.fc4.weight.data
        fc4_bias = model.fc4.bias.data
        fc5_weight = model.fc5.weight.data
        fc5_bias = model.fc5.bias.data
        final_weight = fc5_weight @ fc4_weight @ fc3_weight @ fc2_weight @ fc1_weight
        final_bias = fc5_weight @ fc4_weight @ fc3_weight @ fc2_weight @ fc1_bias \
                    + fc5_weight @ fc4_weight @ fc3_weight @ fc2_bias + \
                    fc5_weight @ fc4_weight @ fc3_bias + \
                    fc5_weight @ fc4_bias + fc5_bias

    log_data.append(f'Final Weights: {final_weight.numpy()}')
    log_data.append(f'Final Biases: {final_bias.numpy()}')

    with torch.no_grad():
        all_val_percentage = model(torch.tensor(X, dtype=torch.float32))

    # Calculate and save overall correlation
    overall_corr = np.corrcoef(all_val_percentage.numpy().flatten(), y)[0, 1]
    log_data.append(f'Overall Correlation: {overall_corr}')

    with open(f'training_log_{gender}.txt', 'w') as f:
        for item in log_data:
            f.write(f"{item}\n")

    plt.scatter(all_val_percentage.numpy(), y, c=percentage, cmap='viridis', alpha=0.6, marker='x', label='Predicted Values')
    plt.xlabel('Predicted 1RM')
    plt.ylabel('Real 1RM')
    plt.title(f'Overall Correlation: {overall_corr:.2f} for {gender}')
    plt.grid(True)
    plt.savefig(f'true_vs_predicted_{gender}.png')
    plt.show()

    # Calculate mean and standard deviation for 1RM and RPE columns based on percentage column
    percentage_groups = data.groupby('percentage').agg(
        mean_1RM=('1RM', 'mean'),
        std_1RM=('1RM', 'std'),
        mean_RPE=('RPE', 'mean'),
        std_RPE=('RPE', 'std')
    ).reset_index()

    percentage_groups.to_csv(f'percentage_summary_{gender}.csv', index=False)

# Read the data
combined_data = pd.read_csv(r'combined-data.csv')

# Divide the data by gender
data_women = combined_data[combined_data['gender'] == 'women']
data_men = combined_data[combined_data['gender'] == 'men']
data_both = combined_data

# Prepare the data for each gender
datasets = {
    'women': data_women,
    'men': data_men,
    'both': data_both
}

# Train the model for each gender
for gender, data in datasets.items():
    X = data[['1RM%', 'RPE']].values
    y = data['1RM'].values
    train_and_evaluate(X, y, gender, data['percentage'])
