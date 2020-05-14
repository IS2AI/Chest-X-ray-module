import torch
import torchvision.transforms as transforms

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(256),
    transforms.Lambda
    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda
    (lambda crops: torch.stack([normalize(crop) for crop in crops]))
])

transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize,
])

transforms_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize,
])
