import torch
import torch.nn as nn
import torch.optim as optim
from models import SimpleModel, AdvancedModel
from datasets import getDigitMNIST
import os
import time

os.system("clear")

def trainScript(EPOCHS=5, LR=0.001, MODEL_TYPE="Simple"):
  train_loader, test_loader = getDigitMNIST()
  
  model = None
  if MODEL_TYPE == "Simple":
    model = SimpleModel()
  elif MODEL_TYPE == "Advanced":
    model = AdvancedModel()
  
  optimizer = optim.Adam(model.parameters(), LR)
  loss_func = nn.CrossEntropyLoss()

  global val_accuracy
  val_accuracy = 0

  print(f"----------------------| {MODEL_TYPE} Training Process Starting... |----------------------\n")
  
  for epoch_num in range(EPOCHS):
    model.train()
    total_loss = 0
    for train_batch_idx, (train_images, train_labels) in enumerate(train_loader):
      train_labels_output = model(train_images)
      train_loss = loss_func(train_labels_output, train_labels)

      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

      total_loss += train_loss.item()

      print(f"\r Epoch {str(epoch_num+1).zfill(2)} | Batch Num {str(train_batch_idx).zfill(4)} | Avg Loss: {(total_loss / (train_batch_idx+1)):.4f}", end='', flush=True)
    
    print()
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for _, (test_images, test_labels) in enumerate(test_loader):
        test_labels_output = model(test_images)
        test_loss = loss_func(test_labels_output, test_labels)
        val_loss += test_loss.item()

        _, predicted = torch.max(test_labels_output, 1)
        correct += (predicted == test_labels).sum().item()
        total += test_labels.size(0)

    avg_val_loss = val_loss / len(test_loader)

    val_accuracy = 100 * correct / total

    print(f"\t\t\t\t\t\tValidation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

  
  WAIT_TIME = 1.5
  COUNT = 4 * 50
  LR_STR = "_".join(str(LR).split("."))
  FILE_NAME = f"Model_Epoch{EPOCHS}_LR{LR_STR}.pth"
  print()
  for i in range(COUNT):
    if i % 4 == 0:
      
      print(f"\r----------------------| Saving \"{FILE_NAME}\"    |----------------------", end="")
    elif i % 4 == 1:
      print(f"\r----------------------| Saving \"{FILE_NAME}\".   |----------------------", end="")
    elif i % 4 == 2:
      print(f"\r----------------------| Saving \"{FILE_NAME}\"..  |----------------------", end="")
    else:
      print(f"\r----------------------| Saving \"{FILE_NAME}\"... |----------------------", end="")
    time.sleep(WAIT_TIME/COUNT)

  torch.save(model.state_dict(), f"./models/{MODEL_TYPE}/{FILE_NAME}")
  print(f"\r----------------------| \"{FILE_NAME}\" Saved! |----------------------")

  return FILE_NAME, model


trainScript(EPOCHS=5, LR=0.001, MODEL_TYPE="Advanced")