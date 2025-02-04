Training a model that detects pneumonia then hosting it on a serverless system

**Key takeaways**:
- Creating a custom dataset class for loading training/testing/validation data
- Finetuning a ResNet18 model (loaded from torchvision and then added a fulling connected layer so it produces 2 outputs) to detect pneumonia (needed to tranform the images first)
    - Training the model on cloud
- Use model weights as handler so we can deploy it as serverless endpoint
- Create a serverless handler/endpoint with AWS lambda