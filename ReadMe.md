# Getting Started

Use the environment.yml file to create the environment needed to run the code

## Load the model to see if it works well like so
```
#Load the model
loaded_model= torch.load('model')

#Write a function to make predictions from the model
def predict_loaded(tensor):
    # Use the loaded model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = loaded_model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor

#Make the predictions from the tensors you want
waveform, sample_rate, utterance, *_ = train_set[10000]
ipd.Audio(waveform.numpy(), rate=sample_rate)
print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")
```

## Training your own model

- It's better to be using a CUDA device for this
- You can check if you're using a cuda device as follows:

```
torch_mem_info = torch.cuda.mem_get_info()
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Globally available:', round(torch_mem_info[0]/1024**3,1), 'GB')
    print('Total:   ', round(torch_mem_info[1]/1024**3,1), 'GB')
```
The output should be as follows (will vary on your own device)

```
Using device: cuda
GPU Model: NVIDIA GeForce RTX 3080
Globally available memory: 9.1 GB
Total GPU memory: 9.8 GB
```

Use the approach_2.py script to get access to the vocabulary dataset for your own training

## Accuracy

Observed accuracy was 67% at 2 epochs and 81% at 20 epochs