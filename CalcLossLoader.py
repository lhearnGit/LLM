from CalcLossBatch import calc_loss_batch
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0: # if no data is loaded, return not a number
        return float("nan")
    elif num_batches is None: # if there are no batches, the batches are equal to the length of the data_loader
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader)) 
        # ensures the number of batches matches the batches in the dataloader if it exceeds it
    for i,(input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches: # for each batch, calculate the loss value and total it
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        return total_loss / num_batches # average loss acrossd all batches
    