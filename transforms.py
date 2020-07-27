import torchvision.transforms as transforms

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])


transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(256),
    transforms.ToTensor(),
    normalize,
])

transforms_val = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    normalize,
])

transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    normalize
])
