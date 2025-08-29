import torch
from torch import cuda, optim, nn, device
from torch.utils.data import DataLoader
from dataset import AnimalImages
from model import get_model
from typing import List, Tuple
from train_config import *
import argparse

@torch.no_grad()
def get_accuracy(arg_model: nn.Module, data_loader: DataLoader, arg_device: device) -> float:
    """ Calculate the accuracy of the model given the data loader """

    arg_model.eval() # Changing the model to eval mode
    correct_predictions, total_predictions = 0, 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(arg_device), labels.to(arg_device)
        outputs = arg_model(imgs)

        # predicted label is one with largest probability, hence finding max
        _, predicted_labels = torch.max(outputs, dim=1)

        # increment the number of correct predictions 
        total_predictions += labels.shape[0]
        correct_predictions += int((predicted_labels == labels).sum())

    return round((correct_predictions / total_predictions) * 100, 2)


def load_state(
    model_name: str, model: nn.Module, optimizer, scheduler) -> None: 
    """ Load the state of the model and its optimizer or scheduler """

    model.load_state_dict(torch.load(f'./data/{model_name}_checkpoint.pth'))
    other_checkpoint = torch.load(f'./data/{model_name}_other_checkpoint.pth')
    optimizer.load(other_checkpoint["optimizer"]) 
    scheduler.load(other_checkpoint["scheduler"])


def save_state(
    model_name: str, model: nn.Module, optimizer, scheduler
) -> None: 
    """ Save the state to checkpoint"""
    torch.save(model.state_dict(), f'/data/{model_name}_checkpoint.pth')
    torch.save(
        {
            "optimizer": optimizer.state_dict(), 
            "scheduler": scheduler.state_dict()
        }, 
        f'./data/{model_name}_other_checkpoint.pth'
    )


def mini_batch_training(
    num_epochs: int, model: nn.Module, train_data_loader: DataLoader, val_data_loader: DataLoader, 
    loss_fn, optimizer, scheduler, 
) -> Tuple[List[float], List[float]]:
    
    """ Training loop over the minibatches of the dataset """
    device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # train loss and val loss
        avg_train_loss, avg_val_loss = 0.0, 0.0

        # compute the train loss
        model.train() # switch the model to train mode
        for batch_ix, (imgs, labels) in enumerate(train_data_loader):
            # make sure the imgs and the labels are converted to be used by right device 
            imgs, labels = imgs.to(device), labels.to(device)
            # forward phase
            predicted_train_labels = model(imgs)
            train_loss = loss_fn(predicted_train_labels, labels)

            # backward phase
            optimizer.zero_grad()
            train_loss.backward() # calculate and accumulate the current gradient
            optimizer.step()

            avg_train_loss += train_loss.item()
            # The user has an option of not using scheduler to train 
            if scheduler: 
                scheduler.step(epoch + batch_ix / len(train_data_loader))
    
        # compute train loss 
        avg_train_loss = round(avg_train_loss / len(train_data_loader), 4)
        train_losses.append(avg_train_loss)

        # compute the val loss, turn off the autograd 
        with torch.no_grad():
            model.eval() # switch the model to val mode
            for imgs, labels in val_data_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                # forward pass
                predicted_val_labels = model(imgs)
                val_loss = loss_fn(predicted_val_labels, labels)
                avg_val_loss += val_loss.item()

        avg_val_loss = round(avg_val_loss / len(val_data_loader), 4)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}/{num_epochs}. Train: {avg_train_loss}, val: {avg_val_loss}")

    # return different tracks for graphing 
    return train_losses, val_losses


def main() -> None: 
    parser = argparse.ArgumentParser(description="Train classification models")

    parser.add_argument("model_name", type=str, help="The name of the model")
    parser.add_argument("--validate", action="store_true", 
        help="Get accuracy of the trained model on the dataset")
    parser.add_argument("--resume", action="store_true", 
        help="Resume training the model instead of from scratch")
    
    model_name: str = parser.parse_args().model_name

    # initialize the dataset 
    train_dataset = AnimalImages("./data/animals10/raw-img", 224)
    val_dataset = AnimalImages("./data/animals10/raw-img", 224, train=False)

    # Data loader of the dataset
    # fix the number of workers based on the machine in which this is trained
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)

    # Initialize the model 
    torch.cuda.empty_cache() 

    # If we only validate the model 
    if parser.parse_args().validate_model: 
        # model with the pretrained parameters 
        animal_classifier = get_model(model_name, load_state=True)
        load_state(model_name, animal_classifier)
        print(get_accuracy(animal_classifier, train_loader), get_accuracy(animal_classifier, val_loader))
        return

    # Newly initialized models 
    animal_classifier = get_model(model_name)
    # If we train the model, use ADAMW to auto update the learning rate
    this_optimizer = optim.AdamW(
        animal_classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGTH_DECAY
    )
    # Scheduler to schedule the global learning rate update periodically
    this_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        this_optimizer,
        T_0=NUM_EPOCHS_EACH_CYCLE, # every cycle is 30 epochs, the scheduler restarts lr
        T_mult=1, # cycle length the same
        eta_min=1e-5 # min LR
    )

    # If training is resumed, load the weights as well as state of the optimizer and scheduler
    if parser.parse_args().resume: 
        load_state(model_name, animal_classifier, this_optimizer, this_scheduler)

    train_track, val_track = mini_batch_training(
        num_epochs=TOTAL_NUM_EPOCHS, 
        model=animal_classifier,
        loss_fn=nn.CrossEntropyLoss(), 
        optimizer=this_optimizer, scheduler=this_scheduler,
        train_data_loader=train_loader, val_data_loader=val_loader
    )
    # save state to checkpoint file  
    save_state(model_name, animal_classifier, this_optimizer, this_scheduler)

if __name__ == "__main__":
    main()