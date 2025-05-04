from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

import telebot
bot = telebot.TeleBot("Bot Token")

np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
@bot.message_handler(content_types=['photo'])
def detected(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

            
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(file_name).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
 
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    bot.send_message(message.chat.id, f'"Относится к:" {class_name[2:]}')
    bot.send_message(message.chat.id, f'"С вероятностью в:" {confidence_score}')

    



bot.infinity_polling()
