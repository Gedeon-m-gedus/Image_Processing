from torchvision import models
my_model = models.resnet101(pretrained=True)

from PIL import Image
import torchvision.transforms.functional as TF

our_image = Image.open('Sabo.jpeg')
our_image_tensor = TF.to_tensor(our_image)

prediction = my_model(our_image_tensor)
print(prediction.argmax().item())
