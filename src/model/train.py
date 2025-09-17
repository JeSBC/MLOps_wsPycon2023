import torch
import torch.nn.functional as F
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader

# Import the model class from the main file
from src.Classifier import Classifier

import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

def train(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
    model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)

            if batch_idx % config.batch_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    batch_idx / len(train_loader), loss.item()))
                
                train_log(loss, example_ct, epoch)

        loss, accuracy = test(model, valid_loader)  
        test_log(loss, accuracy, example_ct, epoch)
    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")

def evaluate(model, test_loader):
    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader)
    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, test_loader, k=32):
    model.eval()
    losses = []
    all_data = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='none')
            pred = output.argmax(dim=1, keepdim=True)
            
            losses.extend(loss.cpu().numpy())
            all_data.extend(data.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    losses = torch.tensor(losses)
    all_data = torch.tensor(all_data)
    all_targets = torch.tensor(all_targets)
    all_predictions = torch.tensor(all_predictions)

    # Ajustar k para que no sea mayor que el número de ejemplos disponibles
    k = min(k, len(losses))
    
    _, indices = torch.topk(losses, k)
    
    highest_k_losses = losses[indices]
    hardest_k_examples = all_data[indices]
    true_labels = all_targets[indices]
    predicted_labels = all_predictions[indices]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels

def train_and_log(config, experiment_id='99'):
    with wandb.init(
        project="Project", 
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="train-model", config=config) as run:
        
        config = wandb.config
        data = run.use_artifact('iris-raw:latest')
        data_dir = data.download()

        training_dataset = read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        
        # Configuración del modelo para Iris
        model_config = {
            "input_shape": 4,  # 4 características en el dataset Iris
            "hidden_layer_1": 128,
            "hidden_layer_2": 64,
            "num_classes": 3  # 3 clases en el dataset Iris
        }
        
        model = Classifier(**model_config)
        model = model.to(device)
 
        train(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained NN model for Iris dataset",
            metadata=model_config)

        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")

        run.log_artifact(model_artifact)

    return model
    
def evaluate_and_log(experiment_id='99', config=None):
    with wandb.init(project="Project", name=f"Eval Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", job_type="eval-model", config=config) as run:
        data = run.use_artifact('iris-raw:latest')
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = DataLoader(testing_set, batch_size=128, shuffle=False)

        model_artifact = run.use_artifact("trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata

        model = Classifier(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        run.summary.update({"loss": loss, "accuracy": accuracy})

        # Crear una tabla para los ejemplos más difíciles
        table = wandb.Table(columns=["features", "true_label", "predicted_label", "loss"])
        for i in range(len(hardest_examples)):
            table.add_data(
                hardest_examples[i].numpy(), 
                int(true_labels[i].numpy()),
                int(preds[i].numpy()),
                highest_losses[i].numpy()
            )
        wandb.log({"hardest_examples": table})

# Training and evaluation loop
epochs = [25, 50, 100]
for id, epoch in enumerate(epochs):
    train_config = {
        "batch_size": 128,
        "epochs": epoch,
        "batch_log_interval": 25,
        "optimizer": "Adam"
    }
    model = train_and_log(train_config, id)
    evaluate_and_log(id)