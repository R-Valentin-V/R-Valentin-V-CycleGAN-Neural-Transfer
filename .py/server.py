from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Функция для загрузки изображения и преобразования его в тензор
def load_image(image_bytes, max_size=400, shape=None):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Изменение размера изображения
    if max_size:
        size = max(image.size)
        if size > max_size:
            size = max_size
        if shape:
            size = shape
        in_transform = transforms.Compose([
            transforms.Resize(600),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    else:
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    # Преобразование изображения в тензор и добавление размерности пакета
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image

# Функция для преобразования тензора обратно в изображение
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

# Определяем модель VGG
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# функция потерь
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Инициализация модели
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG().to(device).eval()

# Определяем архитектуру модели G_AB
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.LeakyReLU(0.2, True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.LeakyReLU(0.2, True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.2, True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

# Параметры
input_nc = 3  # Количество входных каналов
output_nc = 3  # Количество выходных каналов
ngf = 64  # Количество фильтров в генераторе
ndf = 64  # Количество фильтров в дискриминаторе
n_blocks = 12  # Количество резидуальных блоков в генераторе
n_layers_D = 3  # Количество слоев в дискриминаторе
lr = 0.0002  # Скорость обучения
beta1 = 0.1  # Параметр beta1 для оптимизатора Adam

G_AB = Generator(input_nc, output_nc, ngf, n_blocks).to(device)

model_path = 'models/G_AB.pth'
# Загрузка модели на CPU
G_AB.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
G_AB.eval()

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    if 'content_image' in request.files and 'style_image' in request.files:
        print("Выбран обычный")
        content_image_bytes = request.files['content_image'].read()
        style_image_bytes = request.files['style_image'].read()

        content_image = load_image(content_image_bytes, shape=None)
        style_image = load_image(style_image_bytes, shape=content_image.shape[-2:])

        content_image = content_image.to(device)
        style_image = style_image.to(device)

        # Генерация целевого изображения
        target = content_image.clone().requires_grad_(True).to(device)

        # Определяем оптимизатор
        optimizer = optim.Adam([target], lr=0.003)

        # Определяем стиль и вес контента
        style_weight = 1e6
        content_weight = 1

        # Цикл обучения
        for step in range(300):
            target_features = model(target)
            content_features = model(content_image)
            style_features = model(style_image)

            style_loss = content_loss = 0

            for target_feature, content_feature, style_feature in zip(target_features, content_features, style_features):
                content_loss += torch.mean((target_feature - content_feature) ** 2)

                # Вычислить матрицу Грама для стилевых особенностей
                target_gram = gram_matrix(target_feature)
                style_gram = gram_matrix(style_feature)
                style_loss += torch.mean((target_gram - style_gram) ** 2)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step}, Total loss: {total_loss.item()}")

        # Преобразование конечного изображение в PIL-изображение
        final_image = im_convert(target)
        final_image_pil = Image.fromarray((final_image * 255).astype('uint8'))

        # Сохраняем финальное изображение в байтовый массив
        final_image_byte_arr = io.BytesIO()
        final_image_pil.save(final_image_byte_arr, format='PNG')
        final_image_byte_arr.seek(0)

        return jsonify({'image': final_image_byte_arr.getvalue().hex()})

    elif 'content_image' in request.files:
        print("Выбран GAN")
        content_image_bytes = request.files['content_image'].read()
        content_image = load_image(content_image_bytes, shape=None)
        content_image = content_image.to(device)

        # Обработтка изображение с использованием модели GAN
        fake_B = G_AB(content_image)

        # Преобразовать конечное изображение в PIL-изображение
        final_image = im_convert(fake_B)
        final_image_pil = Image.fromarray((final_image * 255).astype('uint8'))

        # Сохраняем финальное изображение в байтовый массив
        final_image_byte_arr = io.BytesIO()
        final_image_pil.save(final_image_byte_arr, format='PNG')
        final_image_byte_arr.seek(0)

        return jsonify({'image': final_image_byte_arr.getvalue().hex()})

    else:
        print("No content_image found in request files")
        return jsonify({'error': 'No image provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
