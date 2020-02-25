from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

rgb_mean = {'cifar10': [0.491, 0.482, 0.447], 'cifar100': [0.507, 0.487, 0.441], 'imagenet': [0.485, 0.456, 0.406]}
rgb_std = {'cifar10': [0.202, 0.199, 0.201], 'cifar100': [0.267, 0.256, 0.276], 'imagenet': [0.229, 0.224, 0.225]}
dataset = {'cifar10': CIFAR10, 'cifar100': CIFAR100, 'imagenet': ImageFolder}


def get_dataset(data_name, data_type='train', data_path=None):
    if data_name == 'imagenet':
        crop_size, color_jitter_factor = [256, 224], 1.0
    else:
        crop_size, color_jitter_factor = [32, 32], 0.5
    if data_type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size[data_name][-1]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8 * color_jitter_factor,
                                                           0.8 * color_jitter_factor,
                                                           0.8 * color_jitter_factor,
                                                           0.2 * color_jitter_factor)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])])
    else:
        transform = transforms.Compose([
            transforms.Resize(crop_size[data_name][0]),
            transforms.CenterCrop(crop_size[data_name][-1]),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean[data_name], rgb_std[data_name])])

    if data_name == 'imagenet':
        return dataset[data_name](root='{}/{}'.format(data_path, 'train' if data_type == 'train' else 'val'),
                                  transform=transform)
    else:
        return dataset[data_name](root='data', train=True if data_type == 'train' else False,
                                  transform=transform, download=True)
