import logging
from telegram import Update, InputFile, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackQueryHandler, CallbackContext
import requests

# Включаем логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Токен вашего бота
TOKEN = ' '

# URL вашего локального сервера Flask
FLASK_SERVER_URL = 'http://localhost:5000/style_transfer'

# Обработчик команды /start
async def start(update: Update, context: CallbackContext) -> None:
    keyboard = [
        [InlineKeyboardButton("GAN", callback_data='1')],
        [InlineKeyboardButton("Базовый вариант", callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message = await update.message.reply_text('Привет! Выберите нейронную сеть:', reply_markup=reply_markup)
    context.user_data['messages'] = [message.message_id]

# Обработчик выбора нейронной сети
async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    context.user_data['selected_network'] = query.data
    if query.data == '1':
        message = await query.edit_message_text(text="Вы выбрали GAN. Отправьте мне одно изображение.")
    elif query.data == '2':
        message = await query.edit_message_text(text="Вы выбрали базовый вариант. Отправьте мне два изображения.")
    context.user_data['messages'].append(message.message_id)

# Обработчик сообщений с изображениями
async def handle_image(update: Update, context: CallbackContext) -> None:
    selected_network = context.user_data.get('selected_network')
    if not selected_network:
        await update.message.reply_text('Пожалуйста, сначала выберите нейронную сеть командой /start.')
        return

    user_data = context.user_data
    if selected_network == '2':
        if 'content_image' not in user_data:
            file = await update.message.photo[-1].get_file()
            user_data['content_image'] = await file.download_as_bytearray()
            await update.message.reply_text('Теперь отправьте style_image.')
        elif 'style_image' not in user_data:
            file = await update.message.photo[-1].get_file()
            user_data['style_image'] = await file.download_as_bytearray()

            response = requests.post(FLASK_SERVER_URL, files={
                'content_image': user_data['content_image'],
                'style_image': user_data['style_image']
            })

            if response.status_code == 200:
                final_image_hex = response.json().get('image')
                final_image_bytes = bytes.fromhex(final_image_hex)
                await update.message.reply_photo(photo=InputFile(final_image_bytes))
            else:
                await update.message.reply_text('Произошла ошибка при обработке изображений.')

            user_data.clear()
    else:
        file = await update.message.photo[-1].get_file()
        image_data = await file.download_as_bytearray()

        response = requests.post(FLASK_SERVER_URL, files={'content_image': image_data})

        if response.status_code == 200:
            final_image_hex = response.json().get('image')
            final_image_bytes = bytes.fromhex(final_image_hex)
            await update.message.reply_photo(photo=InputFile(final_image_bytes))
        else:
            await update.message.reply_text('Произошла ошибка при обработке изображения.')

# Обработчик команды /help
async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Доступные команды:\n/start - Начать взаимодействие с ботом\n/help - Получить помощь\n/clear - Очистить историю\n/change_network - Сменить нейронную сеть')

# Обработчик команды /clear
async def clear_command(update: Update, context: CallbackContext) -> None:
    if 'messages' in context.user_data:
        for message_id in context.user_data['messages']:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message_id)
            except Exception as e:
                logger.error(f"Ошибка при удалении сообщения {message_id}: {e}")
        context.user_data.clear()
    await update.message.reply_text('История взаимодействия очищена.')

# Обработчик команды /change_network
async def change_network(update: Update, context: CallbackContext) -> None:
    keyboard = [
        [InlineKeyboardButton("GAN", callback_data='1')],
        [InlineKeyboardButton("Базовый вариант", callback_data='2')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message = await update.message.reply_text('Выберите новую нейронную сеть:', reply_markup=reply_markup)
    context.user_data['messages'] = [message.message_id]

def main() -> None:
    application = ApplicationBuilder().token(TOKEN).build()

    # Устанавливаем команды меню
    application.bot.set_my_commands([
        BotCommand('start', 'Начать взаимодействие с ботом'),
        BotCommand('help', 'Получить помощь'),
        BotCommand('clear', 'Очистить историю'),
        BotCommand('change_network', 'Сменить нейронную сеть')
    ])

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("change_network", change_network))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()

if __name__ == '__main__':
    main()