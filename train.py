import torch
from torch import cuda, optim, nn
from torch.utils.data import DataLoader
from dataset import AnimalImages
from model.resnet import ResNet101Classifier

@torch.no_grad()
def get_accuracy(arg_model: nn.Module, data_loader: DataLoader, arg_device: cuda.device) -> float:
    """ 
    Calculate the accuracy of the model given the data loader 
    """
    arg_model.eval() # changing the model to eval mode
    correct_prediction, total_prediction = 0, 0
    for imgs, labels in data_loader:
        imgs, labels = imgs.to(arg_device), labels.to(arg_device)
        outputs = arg_model(imgs)

        # predicted label is one with largest probability, hence finding max
        _, predicted_labels = torch.max(outputs, dim=1)
        # increment the number of correct predictions 
        total_prediction += labels.shape[0]
        correct_prediction += int((predicted_labels == labels).sum())

    return round((correct_prediction / total_prediction) * 100, 2)


def mini_batch_training(
    num_epochs: int, model: nn.Module, loss_fn, optimizer, train_data_loader, val_data_loader
):
    """ Training loop over the minibatches of the dataset """
    
    device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # train loss and val loss
        avg_train_loss, avg_val_loss = 0.0, 0.0

        # compute the train loss
        model.train() # switch the model to train mode
        for imgs, labels in train_data_loader:
            # make sure the imgs and the labels
            imgs, labels = imgs.to(device), labels.to(device)

            # forward phase
            predicted_train_labels = model(imgs)
            train_loss = loss_fn(predicted_train_labels, labels)

            # backward phase
            optimizer.zero_grad()
            train_loss.backward() # calculate and accumulate the current gradient
            optimizer.step()

            avg_train_loss += train_loss.item()
    
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

        # Logging in the losses 
        print(f"Epoch {epoch}/{num_epochs}. Train: {avg_train_loss}, val: {avg_val_loss}")

    # return different tracks for graphing 
    return train_losses, val_losses


if __name__ == "__main__":
    # Hyper-parameters
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50 # adjust as needed, recommended as high as 100
    BATCH_SIZE = 128 # adjust as needed 

    # initialize the dataset 
    train_dataset = AnimalImages("/content/animals10/raw-img", 224)
    val_dataset = AnimalImages("/content/animals10/raw-img", 224, train=False)

    # data loader of the dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # initialize the model 
    animal_classifier = ResNet101Classifier().to(
        torch.device("cuda") if cuda.is_available() else torch.device("cpu")
    )
    # use ADAM to auto update the learning rate
    this_optimizer = optim.Adam(animal_classifier.parameters(), lr=LEARNING_RATE)

    # train the model
    torch.cuda.empty_cache() 
    train_track, val_track = mini_batch_training(
        num_epochs=NUM_EPOCHS, 
        model=animal_classifier,
        loss_fn=nn.CrossEntropyLoss(), 
        optimizer=this_optimizer,
        train_data_loader=train_loader, 
        val_data_loader=val_loader
    )

    # You can do something with train_track & val_track, like graphing them
    """
    # Optional, checking the accuracy 
    print(
        get_accuracy(animal_classifier, train_loader),
        get_accuracy(animal_classifier, val_loader)
    )
    """
    # save the model's weight to the file 
    torch.save(animal_classifier.state_dict(), './animals_weight2')