import os
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision import datasets, transforms

from tqdm.auto import tqdm
import sys
from PIL import Image, ImageEnhance, ImageOps

import wandb

ENTITY = 'cs24m504-indian-institute-of-technology-madras-'
PROJECT = 'vgg6-demo-1'

IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if IS_CUDA_AVAILABLE else "cpu")

# Ensure deterministic behavior
def setup_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(hash("Setting seeds manually") % 2**32 - 1)
    np.random.seed(hash("to improve reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("and remove stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so that runs are repeatable") % 2**32 - 1)

# Device configuration
def print_device_details():
    tqdm.write(f"CUDA Available: {IS_CUDA_AVAILABLE}")
    if IS_CUDA_AVAILABLE:
        tqdm.write(f"Number of GPUs available: {torch.cuda.device_count()}")
        tqdm.write(f"Current GPU Name: {torch.cuda.get_device_name(0)}")
        tqdm.write(f"Number of CPU cores available: {os.cpu_count()}")

# Wandb login using API key from file
def wandb_login():
    api_key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wandb_API_Key.txt")
    wandb_api_key = None
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, "r") as f:
                wandb_api_key = f.read().strip()
        except Exception as e:
            tqdm.write(f"Could not read Wandb API key file: {e}")

    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        tqdm.write("Wandb API key not found; falling back to interactive login.")
        wandb.login()

# Custom transformations
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        self.p1 = p1
        self.op1=operation1
        self.magnitude_idx1=magnitude_idx1
        self.p2 = p2
        self.op2=operation2
        self.magnitude_idx2=magnitude_idx2
        self.fillcolor=fillcolor
        self.init = 0

    def gen(self, operation1, magnitude_idx1, operation2, magnitude_idx2, fillcolor):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if self.init == 0:
            self.gen(self.op1, self.magnitude_idx1, self.op2, self.magnitude_idx2, self.fillcolor)
            self.init = 1
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

def GetCifar10(batchsize, val_split=0.1, worker_count=0):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(), # Uncommented data augmentation
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16) # Uncommented data augmentation
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # Load the full training dataset
    full_train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=True)

    # Calculate the size of the validation set
    val_size = int(val_split * len(full_train_data))
    train_size = len(full_train_data) - val_size

    # Split the full training dataset into training and validation sets
    train_data, val_data = torch.utils.data.random_split(full_train_data, [train_size, val_size])

    # Create data loaders for training, validation, and testing
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=worker_count)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=worker_count)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=worker_count)

    return train_dataloader, val_dataloader, test_dataloader

def verify_dataloader_sizes():
    train_loader , val_loader, test_loader = GetCifar10(256)

    # Get the size of the training and test datasets
    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    test_dataset_size = len(test_loader.dataset)

    tqdm.write(f"Size of the training dataset: {train_dataset_size}")
    tqdm.write(f"Size of the validation dataset: {val_dataset_size}")
    tqdm.write(f"Size of the test dataset: {test_dataset_size}")

# VGG Model
class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # fixed output size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def make_layers(layer_cfg, batch_norm=False, activation_function='ReLU'):
    layers = []
    in_channels = 3
    for v in layer_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if activation_function == 'GELU':
                activation = nn.GELU()
            elif activation_function == 'ELU':
                activation = nn.ELU(inplace=True)
            elif activation_function == 'LeakyReLU':
                activation = nn.LeakyReLU(inplace=True)
            elif activation_function == 'PReLU':
                activation = nn.PReLU()
            elif activation_function == 'SiLU':
                activation = nn.SiLU(inplace=True)
            elif activation_function == 'ReLU':
                activation = nn.ReLU(inplace=True)
            else:
                raise ValueError(f"Unsupported activation function: {activation_function}")
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation]
            else:
                layers += [conv2d, activation]
            in_channels = v
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True, activation_function='ReLU'):
    return VGG(make_layers(cfg, batch_norm=batch_norm, activation_function=activation_function), num_classes=num_classes)

def eval(model, data, desc="Evaluation Progress"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(data, desc=desc, leave=False, position=1, dynamic_ncols=True, file=sys.stdout):
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    acc = 100. * correct / total
    return acc

# Train functions
def train_model(model, epochs, optimizer, train_loader, val_loader, criterion):
    # Wrap the epoch loop with tqdm for a progress bar
    for epoch in tqdm(range(epochs), desc="Training Progress", leave=True, position=0, dynamic_ncols=True):
        model.train()  # Set model to training mode at the beginning of each epoch
        running_loss = 0.0
        # Wrap the train_loader with tqdm for a progress bar for batches at position 1
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Batch Progress", leave=False, position=1, dynamic_ncols=True, file=sys.stdout):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate on the validation loader and calculate validation loss
        model.eval()  # Set model to evaluation mode for validation
        val_loss = 0.0
        correct = 0
        total = 0

        # Wrap the val_loader with tqdm for a progress bar for validation batches
        with torch.no_grad():
            # Validation progress bar at position 1 so it displays beneath the epoch bar
            for data, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation Progress", leave=False, position=1, dynamic_ncols=True, file=sys.stdout):
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_acc = 100. * correct / total

        # Evaluate training accuracy (model is still in eval mode from validation)
        train_acc = eval(model, train_loader, f"Epoch {epoch+1}/{epochs} - Training Accuracy Evaluation")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        tqdm.write(f"Epoch {epoch} - Train_Loss: {running_loss/len(train_loader):.4f} , Val_Loss: {val_loss/len(val_loader):.4f} , Train_acc: {train_acc}, Validation_acc : {val_acc}")

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'activation_functions': {
            'values': [ 'GELU', 'ELU', 'LeakyReLU', 'PReLU', 'SiLU']
        },
        'batch_size': {
            'values': [64, 128]
        },
        'optimizers': {
            'values': ['Adam', 'AdamW', 'RMSprop', 'Nesterov-SGD']
        },
        'learning_rate': {
            'values': [0.005, 0.001, 0.0005, 0.0001, 0.00005]
        },
        'epochs': {
            'values': [50, 100]
        }
    },
    'count': 25 # Increased count to allow for more random sampling
}

manual_config = {
    'activation_functions': 'SiLU',
    'batch_size': 64,
    'optimizers': 'Nesterov-SGD',
    'learning_rate': 0.005,
    'epochs': 150
}

def train_and_evaluate_model_for_config():
    config = wandb.config

    # Print the current configuration
    tqdm.write(f"Current run config: {config}")

    # Define a fixed weight decay
    weight_decay_value = 0.0001

    # VGG-6 configuration
    cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']

    # Create the model
    model = vgg(cfg_vgg6, num_classes=10, batch_norm=True, activation_function=config.activation_functions).to(DEVICE)

    # Create data loaders (now returns train, val, and test loaders)
    train_loader, val_loader, test_loader = GetCifar10(config.batch_size)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    if config.optimizers == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay_value)
    elif config.optimizers == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay_value)
    elif config.optimizers == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay_value)
    elif config.optimizers == 'Nesterov-SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay_value)
    elif config.optimizers == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay_value)
    elif config.optimizers == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay_value)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizers}")

    # Train the model
    train_model(model, config.epochs, optimizer, train_loader, val_loader, criterion) # Pass val_loader to train_model for evaluation

    # Optional: Log final test accuracy (using the separate test_loader now)
    final_test_acc = eval(model, test_loader, f"Final Test Accuracy Evaluation")
    wandb.log({"final_test_acc": final_test_acc})

    # Save the entire model locally
    model_path = f"./models/{wandb.run.name}_model.pth"
    torch.save(model, model_path)
    tqdm.write(f"Model saved locally at: {model_path}")

    # Save the model as a wandb artifact
    model_artifact = wandb.Artifact(
        name=f"vgg6-cifar10-model-{wandb.run.name}",
        type="model",
        description="Trained VGG6 model for CIFAR10"
    )
    model_artifact.add_file(model_path)
    try:
        wandb.log_artifact(model_artifact)
        tqdm.write(f"Model artifact logged to wandb with name: vgg6-cifar10-model-{wandb.run.name}")
    except Exception as e:
        tqdm.write(f"Failed to log model artifact to wandb: {e}")

def sweep_train():
    with wandb.init(project=PROJECT, entity=ENTITY):
        train_and_evaluate_model_for_config()
    wandb.finish()

def sweep_train_manual(manual_config):
    with wandb.init(project=PROJECT, entity=ENTITY, config=manual_config):
        tqdm.write("Starting manual test run with config:")
        tqdm.write(str(manual_config))
        train_and_evaluate_model_for_config()
    wandb.finish()

def get_sweep_id():
    # Initialize sweep
    # Ask the user if they want to (c)reate a new sweep or use an (e)xisting sweep
    while True:
        choice = input("Do you want to (c)reate a new sweep or use an (e)xisting sweep? Enter 'c' or 'e': ").lower()
        if choice in ['c', 'e']:
            break
        else:
            tqdm.write("Invalid input. Please enter 'c' or 'e'.")

    if choice == 'c':
        tqdm.write("Creating a new sweep...")
        tqdm.write("Sweep Configuration:")
        tqdm.write(str(sweep_config))
        # Create a new sweep
        try:
            sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)
            tqdm.write(f"New Sweep created with ID: {sweep_id}")
        except Exception as e:
            tqdm.write(f"Error creating sweep: {e}")
            sweep_id = None # Set sweep_id to None if creation fails
    else:
        # Use an existing sweep
        sweep_id = input("Please enter the existing sweep ID: ").strip()
    return sweep_id

def run_sweep_agent(sweep_id, agent_count=1):
    # Ask the user before starting/resuming the sweep agent
    if sweep_id:
        start_agent = input("Do you want to start/resume the wandb agent for this sweep? (y/N): ").strip().lower()
        if start_agent == 'y' or start_agent == 'yes':
            try:
                tqdm.write("Starting/Resuming wandb agent...")
                # Use the python API to start the agent. sweep_train is the function to run for each agent job.
                wandb.agent(sweep_id, function=sweep_train, entity=ENTITY, project=PROJECT, count=agent_count)
            except Exception as e:
                tqdm.write(f"wandb.agent failed: {e}")
        else:
            tqdm.write("wandb agent not started. You can start it later by calling wandb.agent(...)")
    else:
        tqdm.write("Sweep ID not provided. Cannot start wandb agent.")

def main():
    setup_seeds()
    print_device_details()
    wandb_login()
    
    tqdm.write(f"Entity: {ENTITY}")
    tqdm.write(f"Project: {PROJECT}")

    type_of_run = input("Do you want to run a (s)weep or a (m)anual test run? Enter 's' or 'm': ").lower()
    if type_of_run == 'm':
        sweep_train_manual(manual_config)
        return

    sweep_id = get_sweep_id()

    if sweep_id:
        tqdm.write(f"Using Sweep ID: {sweep_id}")
        
        agent_count = input("Enter the number of agents to run for this sweep (default is 1): ").strip()
        agent_count = int(agent_count) if agent_count.isdigit() else 1

        run_sweep_agent(sweep_id, agent_count)
    else:
        tqdm.write("Sweep ID not set. Cannot proceed with running the agent.")

if __name__ == '__main__':
    main()