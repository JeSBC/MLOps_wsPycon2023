import torch
import torchvision
from torch.utils.data import TensorDataset
# Testing
import argparse
import wandb
#Testing

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=0.8):
    """
    Load the Iris dataset and split into train/validation/test sets
    """
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    # First split: separate training set from temporary test set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_size, 
        stratify=y,
        random_state=42
    )

    # Second split: separate validation and test sets from temporary set
    val_test_size = 0.5  # Split temp set equally between val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_test_size,
        stratify=y_temp,
        random_state=42
    )

    # Create TensorDatasets
    training_set = TensorDataset(X_train, y_train)
    validation_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)

    return [training_set, validation_set, test_set]

def load_and_log():
    with wandb.init(
        project="Project",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # This now loads the Iris dataset
        names = ["training", "validation", "test"]

        # Update metadata to reflect Iris dataset
        raw_data = wandb.Artifact(
            "iris-raw", type="dataset",  # Changed name to iris-raw
            description="raw Iris dataset, split into train/val/test",
            metadata={"source": "sklearn.datasets.load_iris",  # Updated source
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        run.log_artifact(raw_data)

# testing
load_and_log()
