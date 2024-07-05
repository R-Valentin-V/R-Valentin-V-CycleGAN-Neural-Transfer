# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и модель
COPY bot.py .
COPY server.py .
COPY models/G_AB.pth models/

# Открываем порт для локального сервера
EXPOSE 5000

# Запускаем бота и сервер
CMD ["sh", "-c", "python bot.py & python server.py"]