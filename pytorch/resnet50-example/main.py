#!/usr/bin/env python
# coding: utf-8

# # Install dependencies

# In[1]:


get_ipython().run_cell_magic('bash', '', 'nvidia-smi\npip install ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org\npip uninstall -y torch torchvision torchaudio\npython -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 torch-tensorrt tensorrt\n')


# # Running the model without optimizations

# In[1]:


import os
import torch
import torchvision
import torch_tensorrt

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True


# In[2]:


# Load pretrained model

resnet50_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
resnet50_model.eval()


# In[3]:


# Download some images

get_ipython().system('mkdir -p ./data')
get_ipython().system('wget  -O ./data/img0.JPG "https://d17fnq9dkz9hgj.cloudfront.net/breed-uploads/2018/08/siberian-husky-detail.jpg?bust=1535566590&width=630"')
get_ipython().system('wget  -O ./data/img1.JPG "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"')
get_ipython().system('wget  -O ./data/img2.JPG "https://www.artis.nl/media/filer_public_thumbnails/filer_public/00/f1/00f1b6db-fbed-4fef-9ab0-84e944ff11f8/chimpansee_amber_r_1920x1080.jpg__1920x1080_q85_subject_location-923%2C365_subsampling-2.jpg"')
get_ipython().system('wget  -O ./data/img3.JPG "https://www.familyhandyman.com/wp-content/uploads/2018/09/How-to-Avoid-Snakes-Slithering-Up-Your-Toilet-shutterstock_780480850.jpg"')

get_ipython().system('wget  -O ./data/imagenet_class_index.json "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"')


# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# 
# Here's a sample execution.

# In[3]:


from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json 

fig, axes = plt.subplots(nrows=2, ncols=2)

for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)      
    plt.subplot(2,2,i+1)
    plt.imshow(img)
    plt.axis('off')

# loading labels    
with open("./data/imagenet_class_index.json") as json_file: 
    d = json.load(json_file)


# Throughout this tutorial, we will be making use of some utility functions; `rn50_preprocess` for preprocessing input images, `predict` to use the model for prediction and `benchmark` to benchmark the inference. 

# In[4]:


# Utility Functions

import numpy as np
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def rn50_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

# decode the results into ([predicted class, description], probability)
def predict(img_path, model, precision="fp32"):
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    if precision == "fp16":
        input_tensor = input_tensor.half()
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)
        
    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)

def benchmark(model, input_shape=(1024, 1, 224, 224), precision='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if precision == 'fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


# With the model downloaded and the util functions written, let's just quickly see some predictions, and benchmark the model in its current un-optimized state.

# In[6]:


for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)
    
    pred, prob = predict(img_path, resnet50_model)
    print('{} - Predicted: {}, Probablility: {}'.format(img_path, pred, prob))

    plt.subplot(2,2,i+1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(pred[1])


# In[7]:


# Model benchmark without Torch-TensorRT
model = resnet50_model.eval().to("cuda")
benchmark(model, input_shape=(128, 3, 224, 224), nruns=100)


# In[9]:


benchmark(model.half(), input_shape=(128, 3, 224, 224), nruns=100, precision="fp16")


# In[10]:


torch.save(model.state_dict(), "resnet50_pretrained.pth")


# # Accelerating with Torch-TensorRT

# ## FP32(single precision)

# In[5]:


model = resnet50_model.eval().to("cuda")

model.load_state_dict(torch.load("resnet50_pretrained.pth"))
model.eval()


# In[12]:


# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP32 precision.
trt_model_fp32 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions = torch.float32, # Run with FP32
    workspace_size = 1 << 22
)


# In[13]:


# Obtain the average time taken by a batch of input
benchmark(trt_model_fp32, input_shape=(128, 3, 224, 224), nruns=100)


# ## FP16 (half precision)

# In[14]:


import torch_tensorrt

# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP16 precision.
trt_model_fp16 = torch_tensorrt.compile(model, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
    enabled_precisions = {torch.half}, # Run with FP16
    workspace_size = 1 << 22
)


# In[15]:


# Obtain the average time taken by a batch of input
benchmark(trt_model_fp16, input_shape=(128, 3, 224, 224), precision='fp16', nruns=100)


# # Compare probabilites

# In[16]:


# Compare probabilities for fp32
for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)
    
    pred, prob = predict(img_path, trt_model_fp32)
    print('{} - Predicted: {}, Probablility: {}'.format(img_path, pred, prob))

    plt.subplot(2,2,i+1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(pred[1])


# In[17]:


# Compare probabilities for fp16
for i in range(4):
    img_path = './data/img%d.JPG'%i
    img = Image.open(img_path)
    
    pred, prob = predict(img_path, trt_model_fp16, precision="fp16")
    print('{} - Predicted: {}, Probablility: {}'.format(img_path, pred, prob))

    plt.subplot(2,2,i+1)
    plt.imshow(img);
    plt.axis('off');
    plt.title(pred[1])


# # Int8 Post Training Quantization (PTQ)

# In[6]:


import torch.utils.data as data
from torchvision import datasets

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Step 1: Create a custom DataLoader for calibration
class CustomCalibrationDataset(data.Dataset):
    def __init__(self, image_folder, transform=None, repeat_factor=1):
        self.dataset = datasets.ImageFolder(image_folder, transform=transform)
        self.repeat_factor = repeat_factor

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]

    def __len__(self):
        return len(self.dataset) * self.repeat_factor


# In[7]:


# Set your image folder path
calib_image_folder = "./calib-data"

# Set the number of images you have for calibration
num_calibration_images = 4
batch_size = 128

# Create a custom calibration dataset that repeats the available images
calib_dataset = CustomCalibrationDataset(calib_image_folder, transform=transform, repeat_factor=batch_size // num_calibration_images)
calib_dataloader = data.DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


# In[8]:


# Step 2: Create the DataLoaderCalibrator
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(calib_dataloader,
                                       use_cache=False,
                                       algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
                                       device=torch.device('cuda:0'))

# Step 3: Define the compile_spec
compile_spec = {
    "inputs": [torch_tensorrt.Input([128, 3, 224, 224])],
    "enabled_precisions": torch.int8,
    "calibrator": calibrator,
    "truncate_long_and_double": True
}

# Step 4: Compile the model with TensorRT
trt_ptq = torch_tensorrt.compile(model, **compile_spec)


# In[10]:


benchmark(trt_ptq, input_shape=(128, 3, 224, 224), nruns=100)


# In[ ]:




