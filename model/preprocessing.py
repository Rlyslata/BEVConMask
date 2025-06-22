import torchvision.transforms as T
_MEAN_PIXEL_IMAGENET = [0.485, 0.456, 0.406]
_STD_PIXEL_IMAGENET = [0.229, 0.224, 0.225]

class Preprocessing:
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self):
        normalize = T.Normalize(mean=_MEAN_PIXEL_IMAGENET, std=_STD_PIXEL_IMAGENET)
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)

