# Стилизация изображений в телеграм-боте 
## CycleGAN + Neural-Transfer             
### stepik ID: 636018238
# 
 # Оглавление

- [Neural-Transfer](#Neural-Transfer)
  - [1. Извлечение признаков](#1-извлечение-признаков)
  - [2. Определение целевого изображения](#2-определение-целевого-изображения)
  - [3. Потери контента](#3-потери-контента)
  - [4. Потери стиля](#4-потери-стиля)
  - [5. Оптимизация](#5-оптимизация)
  - [Пример](#пример)
- [CycleGAN](#CycleGAN)
  - [Для обучения были использованы следующие датасеты](#Для-обучения-были-исользовнны-следующие-датасеты)
  - [1. Генераторы и Дискриминаторы](#1-генераторы-и-дискриминаторы)
  - [2. Циклическая согласованность](#2-циклическая-согласованность)
  - [3. Функции потерь](#3-функции-потерь)
  - [4. Обучение](#4-обучение)
  - [Пример](#примеры)
- [TG bots](#tg-bots)
- [Server](#Server)
- [Docker](#Docker)  
   - [tar](#tar)      
   - [creations](#creations)
#
   
# [Neural-Transfer](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/blob/main/.ipynb/VGG%20Gram_matrix.ipynb)        
### 1. Извлечение признаков: 
Используется предварительно обученная модель VGG19 для извлечения признаков из изображений.  

### 2. Определение целевого изображения:

Целевое изображение инициализируется как копия изображения контента и оптимизируется в процессе обучения.

### 3. Потери контента:

Потери контента измеряют, насколько хорошо целевое изображение сохраняет контент исходного изображения. Это делается путем сравнения признаков, извлеченных из целевого и контентного изображений. Вычисляется разность между соответствующими тензорами признаков целевого и контентного изображений.

$$
L_{\text{content}} = \sum_{i} \frac{1}{N_i} \sum_{j} (F_{ij} - P_{ij})^2
$$

### 4. Потери стиля:

Потери стиля измеряют, насколько хорошо целевое изображение соответствует стилю стилевого изображения. Это делается путем сравнения матриц Грама (Gram matrices), которые характеризуют корреляции между различными признаками в слоях сети.        
Матрица Грама G для набора признаков 𝐹

$$
G_{ij} = \sum_{k} F_{ik} F_{jk}
$$

### 5. Оптимизация:

Целевое изображение оптимизируется с использованием градиентного спуска, где общая потеря является суммой взвешенных потерь контента и стиля. Оптимизация направлена на минимизацию этой общей потери, что приводит к изображению, которое сочетает в себе контент исходного изображения и стиль другого изображения.

## Пример 
![Input 1](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/raw/main/Images/input%201.png)
<p align="center">
  <img src="https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/blob/main/Images/output_image%201.png" alt="output_image 1.png">
</p>

#
      

# [CycleGAN](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/blob/main/.ipynb/CycleGAN_.ipynb)    
|[weights](https://drive.google.com/file/d/1nphc9T5y1GL74TnQlK9OgtR4G505vW59/view?usp=drive_link)|

CycleGAN (Cycle-Consistent Adversarial Networks) — это тип генеративно-состязательной сети (GAN), которая позволяет выполнять задачи трансляции изображений между двумя немаркированными доменами без использования парных данных. Основной принцип CycleGAN заключается в том, чтобы научиться преобразовывать изображения из одного домена в другой, сохраняя при этом циклическую согласованность (cycle consistency).
### Для обучения были исользовнны следующие датасеты:    

   -    [Flickr8k-600](https://www.kaggle.com/datasets/xxxcccwork/flickr8k600)
   -    [MaySpace](https://drive.google.com/file/d/1HfLbigm6kblWgEtabgzlym3-DO9PfWwi/view?usp=drive_link)

### 1. Генераторы и Дискриминаторы:    
 - **Генераторы:** CycleGAN использует два генератора, A и B. Генератор G учится преобразовывать изображения из домена X в домен Y, а генератор B — из домена Y в домен X.
 -  **Дискриминаторы:** Также используются два дискриминатора, $D_x$ и $D_y$ . Дискриминатор $D_x$ пытается отличить реальные изображения из домена X от сгенерированных генератором B, а дискриминатор $D_y$  — реальные изображения из домена Y от сгенерированных генератором A.

### 2. Циклическая согласованность:

- **Прямая циклическая согласованность:** Если изображение из домена X преобразовано в домен Y с помощью генератора A, а затем обратно в домен X с помощью генератора B, результирующее изображение должно быть близко к исходному. Это выражается как B(A(𝑥))≈𝑥 
- **Обратная циклическая согласованность:** Аналогично, если изображение из домена Y преобразовано в домен X с помощью генератора B, а затем обратно в домен Y с помощью генератора A, результирующее изображение должно быть близко к исходному. Это выражается как 
A(B(y))≈y.
### 3. Функции потерь:     

- **GAN Loss (`criterion_GAN`):**
   $$\[ \text{MSELoss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]$$

- **Cycle Consistency Loss (`criterion_cycle`), Identity Loss (`criterion_identity`):**
    $$\[ \text{L1Loss} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \] $$


   
- **Style Loss (`loss_G_style`), content loss (`loss_G_content `):**    
   $$G_{ij} = \sum_{k} F_{ik} F_{jk}$$

Слои извлечения VGG:
- Style Loss - [ '1', '5', '10', '14', '19', '22', '25', '28', '32' ] 
- content loss - [ '21']


### 4. Обучение:
 - CycleGAN обучается с использованием мини-батчей изображений из двух доменов. Генераторы и дискриминаторы обновляются итеративно, чтобы минимизировать соответствующие функции потерь.
   
## Примеры
<p align="center">
  <img src="https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/blob/main/Images/output_image%205.png" alt=output_image 4.png">
</p>       
 
#
#  [TG bots](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/blob/main/.py/bot.py)       

<p align="center">
  <a href="https://www.youtube.com/watch?v=ybaHsG_J9lw">
    <img src="https://img.youtube.com/vi/ybaHsG_J9lw/0.jpg" alt="Видео на YouTube">
  </a>
</p>

#
 
#  [Server](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/blob/main/.py/server.py)    


#  [Docker](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/tree/main/Docker)     
### |[tar](https://drive.google.com/file/d/17po7O6MhQBRp60_kssxTZz2tujUSsmtN/view?usp=drive_link)| 
### |[creations](https://github.com/R-Valentin-V/R-Valentin-V-CycleGAN-Neural-Transfer/tree/main/Docker/creations)|
