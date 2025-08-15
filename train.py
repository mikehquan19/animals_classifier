import torch
from torch import cuda, optim, nn
from torch.utils.data import DataLoader
from dataset import AnimalImages
from model.resnet import ResNet50Classifier

@torch.no_grad()
def get_accuracy(arg_model: nn.Module, data_loader: DataLoader, arg_device) -> float:
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


def mini_batch_training(
    num_epochs: int, model: nn.Module, loss_fn, optimizer, scheduler, 
    train_data_loader: DataLoader, val_data_loader: DataLoader
):
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
                scheduler.step(epoch + batch_ix / len(train_data_loader)) # updates the global LR every batch
    
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
    """ This current hyper-parameters achieved 80.7% -> 81.2% val accuracy for Resnet-50 """
    # Hyper-parameters
    LEARNING_RATE = 1e-4
    WEIGTH_DECAY = 1e-4
    TOTAL_NUM_EPOCHS = 200 # adjust as needed, recommended as high as 200
    NUM_EPOCHS_EACH_CYCLE = 30
    BATCH_SIZE = 128 # adjust as needed 
    FIRST_ITERATION = True # used in case you resume training

    # initialize the dataset 
    train_dataset = AnimalImages("./data/animals10/raw-img", 224)
    val_dataset = AnimalImages("./data/animals10/raw-img", 224, train=False)
    assert len(train_dataset) + len(val_dataset) == 26179

    # data loader of the dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # initialize the model 
    animal_classifier = ResNet50Classifier().to(
        torch.device("cuda") if cuda.is_available() else torch.device("cpu"))
    
    # use ADAMW to auto update the learning rate (Could also use Adam instead, or even SGD with momentum tbh)
    this_optimizer = optim.AdamW(animal_classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGTH_DECAY)

    # Scheduler to schedule the global learning rate update periodically
    this_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        this_optimizer,
        T_0=NUM_EPOCHS_EACH_CYCLE, # every cycle is 30 epochs, the scheduler restarts lr
        T_mult=1, # cycle length the same
        eta_min=1e-5 # min LR
    )

    # Load the weights as well as state of model and optimizer and schedulers
    if not FIRST_ITERATION: 
        checkpoint = torch.load('./data/animals_checkpoint.pth')

        animal_classifier.load_state_dict(checkpoint['model'])
        this_optimizer.load_state_dict(checkpoint['optimizer'])
        this_scheduler.load_state_dict(checkpoint['scheduler'])

    # train the model
    torch.cuda.empty_cache() 
    train_track, val_track = mini_batch_training(
        num_epochs=TOTAL_NUM_EPOCHS, 
        model=animal_classifier,
        loss_fn=nn.CrossEntropyLoss(), 
        optimizer=this_optimizer,
        scheduler=this_scheduler,
        train_data_loader=train_loader, 
        val_data_loader=val_loader
    )

    # You can do something with train_track & val_track, like graphing them
    torch.cuda.empty_cache()
    print(get_accuracy(animal_classifier, train_loader), get_accuracy(animal_classifier, val_loader))

    # save the model's weight to the file 
    torch.save({
        "model": animal_classifier.state_dict(),
        "optimizer": this_optimizer.state_dict(),
        "scheduler": this_scheduler.state_dict()
    }, './data/animals_checkpoint.pth')