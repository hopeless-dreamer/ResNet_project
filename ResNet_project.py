import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st 
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import requests
from io import BytesIO
import time
import gdown
import os

url_model = "https://drive.google.com/file/d/14HfVKmqASLDfENbJOwPQMquS7_Judx_t/view?usp=drive_link"
model1 = "model.pth"
if not os.path.exists(model1):
    gdown.download(url_model, model1, quiet=False)

url_best_model = "https://drive.google.com/file/d/1AWKALYW74yUP3vT5iNzyzT21VTAsllHR/view?usp=drive_link"
model2 = "best_model_blood_cells.pt"
if not os.path.exists(model2):
    gdown.download(url_best_model, model2, quiet=False)

page = st.sidebar.selectbox("Выберите страницу", ["Спортивная фотография", "Клетки крови"])

if page == "Спортивная фотография":
    st.title(":grey[Определение вида спорта по фото]")
    st.subheader('**:blue[Загрузите любую фотку — получите ML-обработку!]**')
    img = st.file_uploader(':orange[Загрузите ваше фото]', type=['jpg','jpeg','png'])
    resize = T.Resize((224, 224))
    trnsfrms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
    ])
    list_classes=['air hockey',
    'ampute football',
    'archery',
    'arm wrestling',
    'axe throwing',
    'balance beam',
    'barell racing',
    'baseball',
    'basketball',
    'baton twirling',
    'bike polo',
    'billiards',
    'bmx',
    'bobsled',
    'bowling',
    'boxing',
    'bull riding',
    'bungee jumping',
    'canoe slamon',
    'cheerleading',
    'chuckwagon racing',
    'cricket',
    'croquet',
    'curling',
    'disc golf',
    'fencing',
    'field hockey',
    'figure skating men',
    'figure skating pairs',
    'figure skating women',
    'fly fishing',
    'football',
    'formula 1 racing',
    'frisbee',
    'gaga',
    'giant slalom',
    'golf',
    'hammer throw',
    'hang gliding',
    'harness racing',
    'high jump',
    'hockey',
    'horse jumping',
    'horse racing',
    'horseshoe pitching',
    'hurdles',
    'hydroplane racing',
    'ice climbing',
    'ice yachting',
    'jai alai',
    'javelin',
    'jousting',
    'judo',
    'lacrosse',
    'log rolling',
    'luge',
    'motorcycle racing',
    'mushing',
    'nascar racing',
    'olympic wrestling',
    'parallel bar',
    'pole climbing',
    'pole dancing',
    'pole vault',
    'polo',
    'pommel horse',
    'rings',
    'rock climbing',
    'roller derby',
    'rollerblade racing',
    'rowing',
    'rugby',
    'sailboat racing',
    'shot put',
    'shuffleboard',
    'sidecar racing',
    'ski jumping',
    'sky surfing',
    'skydiving',
    'snow boarding',
    'snowmobile racing',
    'speed skating',
    'steer wrestling',
    'sumo wrestling',
    'surfing',
    'swimming',
    'table tennis',
    'tennis',
    'track bicycle',
    'trapeze',
    'tug of war',
    'ultimate',
    'uneven bars',
    'volleyball',
    'water cycling',
    'water polo',
    'weightlifting',
    'wheelchair basketball',
    'wheelchair racing',
    'wingsuit flying']
    try:
        img = Image.open(img)
        st.image(img, channels='RGB')
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 100)  
        model.load_state_dict(torch.load(model1, map_location=torch.device('cpu')))
        model.eval()
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
            ])
        img = transform(img).unsqueeze(0)
        with torch.inference_mode():
            pred_class = model(img).argmax(dim=1).item()  
        result = list_classes[pred_class]
        st.subheader(f'На фотке {result} :wink:')    
    except:
        st.stop()
else:
    # Определение параметров модели
    num_classes = 4  # Замените на количество классов вашей задачи

    # Создание экземпляра модели ResNet-50
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Переопределяем последний слой
    model.load_state_dict(torch.load(model2, map_location=torch.device('cpu')))
    model.eval()  # Переводим модель в режим оценки

    # Определение преобразований изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    st.title("Классификация изображений клеток")

    # Создание полей для загрузки изображений
    uploaded_files = st.file_uploader("Выберите изображение клетки", type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True)

    # Поле для ввода URL
    url = st.text_input("Или введите URL изображения:")


    # Обработка загруженных изображений
    def process_image(image):
        image = transform(image).unsqueeze(0)  # Применяем преобразования и добавляем размерность для батча
        return image


    if uploaded_files or url:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Загруженное изображение',
                        use_column_width=True)  # Отображаем загруженное изображение
                image_tensor = process_image(image)

                start_time = time.time()  # Запоминаем время начала
                with torch.inference_mode():
                    logits = model(image_tensor)  # Получаем логиты от модели
                    preds = torch.softmax(logits, dim=1)  # Применяем Softmax для получения вероятностей
                    pred_class = torch.argmax(preds, dim=1).item()  # Получаем предсказанный класс
                end_time = time.time()  # Запоминаем время окончания

                # Определение меток классов
                class_labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]  # Замените на свои классы
                st.write("Класс:", class_labels[pred_class])
                st.write("Время обработки: {:.2f} сек.".format(end_time - start_time))  # Вывод времени обработки

        # Обработка изображения из URL
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption='Изображение из URL', use_column_width=True)  # Отображаем изображение из URL
                image_tensor = process_image(image)

                start_time = time.time()  # Запоминаем время начала
                with torch.inference_mode():
                    logits = model(image_tensor)  # Получаем логиты от модели
                    preds = torch.softmax(logits, dim=1)  # Применяем Softmax для получения вероятностей
                    pred_class = torch.argmax(preds, dim=1).item()  # Получаем предсказанный класс
                end_time = time.time()  # Запоминаем время окончания

                # Определение меток классов
                class_labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]  # Замените на свои классы
                st.write("Класс:", class_labels[pred_class])
                st.write("Время обработки: {:.2f} сек.".format(end_time - start_time))  # Вывод времени обработки

            except Exception as e:
                st.error("Ошибка при загрузке изображения из URL: {}".format(e))

