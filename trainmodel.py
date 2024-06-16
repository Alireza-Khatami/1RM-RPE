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
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x

def scale_pd(df):
    scaler_params = {}
    scaled_df = pd.DataFrame()
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:  # Ensure the column is numeric
            mean = df[column].mean()
            std = df[column].std()
            scaler_params[column] = {'mean': mean, 'std': std}
            scaled_df[column] = (df[column] - mean) / std
        else:
            scaled_df[column] = df[column]  # Keep non-numeric columns unchanged
    return scaled_df.values, scaler_params
def scale_np(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    scaled_arr = (arr - mean) / std
    scaler_params = {'mean': mean, 'std': std}
    return scaled_arr.values, scaler_params

# Function to train the model and log errors
def train_and_evaluate(X, y, gender,percentage):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    input_size = X.shape[1]
    output_size = 1
    X, x_params = scale_pd(X)
    y,y_params = scale_np(y)
    print (f"X params scaler {x_params}")
    print (f"Y params scaler {y_params}")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = RPEModel(input_size, output_size)
        # Standardize the data
        # x_scaler = StandardScaler()
        # X_train = x_scaler.fit_transform(X_train)
        # X_val = x_scaler.transform(X_val)
        # y_scaler = StandardScaler()
        # y_train = y_scaler.fit_transform(y_train)
        # y_val = y_scaler.transform(y_val)


        # Save the scaler
        # with open(f'x_scaler_{gender}_fold_{fold+1}.pkl', 'wb') as f:
        #     pickle.dump(x_scaler, f)
        # with open(f'y_scaler_{gender}_fold_{fold+1}.pkl', 'wb') as f:
        #     pickle.dump(y_scaler, f)
        # Convert to PyTorch tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        # Training
        for epoch in range(100):
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


        # Save training log
        with open(f'training_log_{gender}_fold_{fold+1}.txt', 'w') as f:
            f.write(f'Fold {fold+1} - Validation Loss: {val_loss}\n')

    avg_val_loss = np.mean(fold_results)
    print(f'Average Validation Loss for {gender}: {avg_val_loss}')

    # Compute the final matrix product of the network's weights
    # with torch.no_grad():
    #     for param in model.parameters():
    #         if final_weights is None:
    #             final_weights = param.data
    #         else:
    #             final_weights = final_weights @ param.data.T
            
    with torch.no_grad():
    # Extract weights and biases
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
        # Compute the combined weights and biases
        final_weight = fc5_weight@fc4_weight@fc3_weight @ fc2_weight @ fc1_weight
        final_bias = fc5_weight@fc4_weight@fc3_weight @ fc2_weight @ fc1_bias \
                    + fc5_weight@fc4_weight@fc3_weight @ fc2_bias + \
                    fc5_weight@fc4_weight @ fc3_bias +\
                    fc5_weight@+fc4_bias + fc5_bias


    # layers = [model.fc1, model.fc2, model.fc3, model.fc4, model.fc5]
    # with torch.no_grad():
    #     # Initialize final weights and biases
    #     final_weight = layers[-1].weight.data
    #     final_bias = layers[-1].bias.data

    #     # Iterate over the layers in reverse order (excluding the last layer)
    #     for layer in reversed(layers[:-1]):
    #         final_weight = final_weight @ layer.weight.data
    #         final_bias = final_bias + final_weight @ layer.bias.data

    # return final_weight, final_bias
    # Save the final matrix to a file
    np.savetxt(f'final_weights_{gender}.txt', final_weight.numpy())
    np.savetxt(f'final_biases_{gender}.txt', final_bias.numpy())
    with torch.no_grad():
        all_val_percentage = model(X_t)
    plt.scatter(all_val_percentage.numpy(), y, c=percentage, cmap='viridis', alpha=0.6, marker='x', label='Predicted Values')
    plt.xlabel('predicted 1RM')
    plt.ylabel('real 1RM')
    # plt.xlim(left=0)  # Limit x-axis to values greater than 0
    # plt.ylim(bottom=0)  # Limit y-axis to values greater than 0 
    # plt.legend()
    plt.grid(True)
    plt.savefig(f'true_vs_predicted_{gender}.png')
    plt.show()

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
    X = data[['1RM%', 'RPE']]
    y = data['1RM']
    train_and_evaluate(X, y, gender,data['percentage'] )
