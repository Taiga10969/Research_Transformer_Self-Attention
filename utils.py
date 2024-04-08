import torch
from tqdm import tqdm

#　精度計算
def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    correct = (max_preds.squeeze(1)==y).float()
    return correct.sum() / len(y)


def train(model, dataloader, tokenizer, optimizer, criterion, scheduler, args, wandb):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    device = next(model.parameters()).device
    
    dataloader = tqdm(dataloader, desc="Training")
    for batch_idx, (text, label) in enumerate(dataloader):
        optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory 
        
        label = label.to(device)
        
        inputs = tokenizer(text, 
                           padding=True,
                           truncation=True, 
                           max_length=args.max_seq_length, 
                           return_tensors="pt", 
                           return_attention_mask=True,
                           )
        
        inputs = {key: val.to(device) for key, val in inputs.items()} 
        
        predictions = model(inputs)

        loss = criterion(predictions, label)
        acc = categorical_accuracy(predictions, label)

        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        dataloader.set_postfix_str(f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

        if args.wandb:
            wandb.log({
                "train_iter_loss" : loss.item(),
                "lr" : optimizer.param_groups[0]['lr'],
                })

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)



def evaluate(model, dataloader, tokenizer, criterion, args, wandb):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    device = next(model.parameters()).device
    
    dataloader = tqdm(dataloader, desc="Evaluate")
    with torch.no_grad():
        for batch_idx, (text, label) in enumerate(dataloader):
            torch.cuda.empty_cache() # releases all unoccupied cached memory 

            label = label.to(device)
            
            inputs = tokenizer(text, 
                               padding=True,
                               truncation=True, 
                               max_length=args.max_seq_length, 
                               return_tensors="pt", 
                               return_attention_mask=True,
                               )
            inputs = {key: val.to(device) for key, val in inputs.items()} 

            predictions = model(inputs)
    
            loss = criterion(predictions, label)
            acc = categorical_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            dataloader.set_postfix_str(f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

            if args.wandb:
                wandb.log({
                    "valed_iter_loss" : loss.item(),
                    })
            
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)
