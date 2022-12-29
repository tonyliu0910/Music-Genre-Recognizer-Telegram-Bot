"""
The following code is modified from https://github.com/pragnakalp/telegram-bot-python to meet our specs.
"""


from flask import Flask, request, Response
import requests
import json
import os
from pytube import YouTube
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pydub
from PIL import Image
from pydub import AudioSegment
from moviepy.editor import *
from lib.language_classifier import *
from lib.efficientnet import *

# token that we get from the BotFather
TOKEN = " {Your Telegram API Token} "

device = get_default_device()
print(device)

efnb0 = EfficientNet.from_name("efficientnet-b0")

# for param in efnb0.parameters():
#   param.requires_grad = False

efnb0._fc = nn.Sequential(
    nn.Linear(in_features=1280, out_features=320),
    nn.ReLU(),
    nn.Linear(in_features=320, out_features=80),
    nn.ReLU(),
    nn.Linear(in_features=80, out_features=15),
)
model = to_device(EfficientNetB0_cifar100(efnb0), device)
load_pretrained_weights(model, "efficientnet-b0", weights_path="weights/EfficientNet-b0.pth")


app = Flask(__name__)

def classify(music, model):
    print("convert mp3 to wav")
    sound = AudioSegment.from_mp3(music)
    sound.export('temp.wav', format="wav")
    print("convert wav to jpg")
    plt.axis("off")
    plt.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])
    x, sr = librosa.load('temp.wav', sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
    plt.savefig('image/temp/temp.jpg',bbox_inches=None,pad_inches=0,)
    plt.close()
    valid_tfms = tt.Compose([tt.ToTensor()])
    valid_ds = ImageFolder("image", valid_tfms)
    valid_dl = DataLoader(valid_ds, 1)
    valid_dl = DeviceDataLoader(valid_dl, device)
    print("predicting genre")
    model.eval()
    genre_list =  ['Pop', 'Reggae', 'Electronic', 'Punk', 'Jazz', 'World', 'Rap', 'Folk', 'Latin', 'New Age', 'Blues', 'Metal', 'Country', 'Rock', 'RnB']
    for i, (data, target) in enumerate(valid_dl):
        data, target = data.to(device), target.to(device)
        output = model(data)
        print(output)
        pred = output.data.max(1, keepdim=True)[1]
        print(pred)
        for j in range(pred.size()[0]):
            pred = genre_list[pred[j].cpu().numpy()[0]]
    return pred


# Reading the JSON message when the user send any type of file to the bot and extracting the chat id of the user and the file id for the file that user send to the bot
def tel_parse_get_message(message):
    print("message-->", message)

    try:  # if the file is an image
        g_chat_id = message['message']['chat']['id']
        g_file_id = message['message']['photo'][0]['file_id']
        print("g_chat_id-->", g_chat_id)
        print("g_image_id-->", g_file_id)

        return g_file_id, g_chat_id
    except:
        try:  # if the file is a video
            g_chat_id = message['message']['chat']['id']
            g_file_id = message['message']['video']['file_id']
            print("g_chat_id-->", g_chat_id)
            print("g_video_id-->", g_file_id)

            return g_file_id, g_chat_id
        except:
            try:  # if the file is an audio
                g_chat_id = message['message']['chat']['id']
                g_file_id = message['message']['audio']['file_id']
                print("g_chat_id-->", g_chat_id)
                print("g_audio_id-->", g_file_id)

                return g_file_id, g_chat_id
            except:
                try:  # if the file is a document
                    g_chat_id = message['message']['chat']['id']
                    g_file_id = message['message']['document']['file_id']
                    print("g_chat_id-->", g_chat_id)
                    print("g_file_id-->", g_file_id)

                    return g_file_id, g_chat_id
                except:
                    print("NO file found found-->>")


# Reading the JSON format when we send the text message and extracting the chat id of the user and the text that user send to the bot
def tel_parse_message(message):
    print("message-->", message)
    try:
        chat_id = message['message']['chat']['id']
        txt = message['message']['text']
        print("chat_id-->", chat_id)
        print("txt-->", txt)

        return chat_id, txt
    except:
        print("NO text found-->>")

    try:
        chat_id = message['callback_query']['from']['id']
        i_txt = message['callback_query']['data']
        print("chat_id-->", chat_id)
        print("i_txt-->", i_txt)

        return chat_id, i_txt
    except:
        pass


# Get the Text message response from the bot
def tel_send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    r = requests.post(url, json=payload)
    return r


# Get the Image response from the bot by providing the image link
# def tel_send_image(chat_id):
#     url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'
#     payload = {
#         'chat_id': chat_id,
#         'photo': "https://raw.githubusercontent.com/fbsamples/original-coast-clothing/main/public/styles/male-work.jpg"
#     }
#     r = requests.post(url, json=payload)
#     return r


# Get the Poll response from the bot
def tel_send_poll(chat_id):
    url = f'https://api.telegram.org/bot{TOKEN}/sendPoll'
    payload = {
        'chat_id': chat_id,
        "question": "In which direction does the sun rise?",
        # options are provided in json format
        "options": json.dumps(["North", "South", "East", "West"]),
        "is_anonymous": False,
        "type": "quiz",
        # Here we are providing the index for the correct option(i.e. indexing starts from 0)
        "correct_option_id": 2
    }
    r = requests.post(url, json=payload)
    return r


# # Get the Button response in the keyboard section
# def tel_send_button(chat_id):
#     url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'

#     payload = {
#         'chat_id': chat_id,
#         'text': "What is this?",    # button should be in the propper format as described
#         'reply_markup': {
#             'keyboard': [[
#                 {
#                     'text': 'supa'
#                 },
#                 {
#                     'text': 'mario'
#                 }
#             ]]
#         }
#     }
#     r = requests.post(url, json=payload)
#     return r


# Get the Inline button response
def tel_send_inlinebutton(chat_id):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'

    payload = {
        'chat_id': chat_id,
        'text': "Share your music!",
        'reply_markup': {
            "inline_keyboard": [[
                {
                    "text": "URL",
                    "callback_data": "ic_A"
                },
                {
                    "text": "Attachment",
                    "callback_data": "ic_B"
                }]
            ]
        }
    }
    r = requests.post(url, json=payload)
    return r


# Get the Button response from the bot with the redirected URL
def tel_send_inlineurl(chat_id):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'

    payload = {
        'chat_id': chat_id,
        'text': "What do you want to know about",
        'reply_markup': {
            "inline_keyboard": [
                [
                    {"text": "Dataset", "url": "https://colinraffel.com/projects/lmd/"},
                    {"text": "Model", "url": "https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html"}
                ]
            ]
        }
    }
    r = requests.post(url, json=payload)
    return r


# Get the Audio response from the bot by providing the URL for the audio
def tel_send_audio(chat_id, file_path):
    url = f'https://api.telegram.org/bot{TOKEN}/sendAudio'

    payload = {
        'chat_id': chat_id,
        "audio": file_path,
    }
    r = requests.post(url, json=payload)
    return r

def tel_send_image(chat_id, image_path):
    url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'
    payload = {
        'chat_id': chat_id,
        'parse_mode': 'HTML'
    }
    files = {
        'photo':open(image_path, 'rb')
    }
    r = requests.post(url, data=payload, files=files)
    return r

def tel_send_genre_image(chat_id, genre):
    url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'
    payload = {
        'chat_id': chat_id,
        'parse_mode': 'HTML'
    }
    if genre == "Pop":
        files = {'photo':open("cool/pop.png", 'rb')}
    elif genre == "Reggae":
        files = {'photo':open("cool/reggae.png", 'rb')}
    elif genre == "Electronic":
        files = {'photo':open("cool/electonic.jpg", 'rb')}
    elif genre == "Punk":
        files = {'photo':open("cool/punk", 'rb')}
    elif genre == "Jazz":
        files = {'photo':open("cool/jazz.png", 'rb')}
    elif genre == "World":
        files = {'photo':open("cool/world.png", 'rb')}
    elif genre == "Rap":
        files = {'photo':open("cool/rap.png", 'rb')}
    elif genre == "Folk":
        files = {'photo':open("cool/folk.png", 'rb')}
    elif genre == "Latin":
        files = {'photo':open("cool/latin.png", 'rb')}
    elif genre == "New Age":
        files = {'photo':open("cool/new age.png", 'rb')}
    elif genre == "Blues":
        files = {'photo':open("cool/blues.png", 'rb')}
    elif genre == "Metal":
        files = {'photo':open("cool/metal.png", 'rb')}
    elif genre == "Country":
        files = {'photo':open("cool/country.png", 'rb')}
    elif genre == "Rock":
        files = {'photo':open("cool/rock.png", 'rb')}
    elif genre == "RnB":
        files = {'photo':open("cool/rnb.png", 'rb')}

    r = requests.post(url, data=payload, files=files)
    return r


def tel_send_document(chat_id):
    url = f'https://api.telegram.org/bot{TOKEN}/sendDocument'
    file_location = 'documents/EfficientNet.pdf'
    payload = {
        'chat_id': chat_id,
        'parse_mode': 'HTML',
        'caption': 'You can learn more from this'
    }
    files = {
        'document':open(file_location, 'rb')
    }
    r = requests.post(url, data=payload, files=files, stream=True)
    return r

# Get the Video response from the bot by providing the URL for the Video
def tel_send_video(chat_id):
    url = f'https://api.telegram.org/bot{TOKEN}/sendVideo'

    payload = {
        'chat_id': chat_id,
        "video": "https://www.appsloveworld.com/wp-content/uploads/2018/10/640.mp4",
    }
    r = requests.post(url, json=payload)
    return r


# Get the url for the file through the file id
def tel_upload_file(file_id):
    # Getting the url for the file
    url = f'https://api.telegram.org/bot{TOKEN}/getFile?file_id={file_id}'
    a = requests.post(url)
    json_resp = json.loads(a.content)
    print("json_resp-->", json_resp)
    file_pathh = json_resp['result']['file_path']
    print("file_pathh-->", file_pathh)

    # saving the file to our computer
    url_1 = f'https://api.telegram.org/file/bot{TOKEN}/{file_pathh}'
    b = requests.get(url_1)
    file_content = b.content
    with open(file_pathh, "wb") as f:
        f.write(file_content)
    return file_pathh
    


# Reading the respomnse from the user and responding to it accordingly
@app.route('/', methods=['GET', 'POST'])
def index():
    music_file_path = ""
    status = 0
    if request.method == 'POST':
        msg = request.get_json()
        try:
            chat_id, txt = tel_parse_message(msg)
            if txt == "hi":
                tel_send_message(chat_id, "Hello, world!")
            elif txt == "image":
                tel_send_image(chat_id)
            elif txt == "poll":
                tel_send_poll(chat_id)
            elif txt == "button":
                tel_send_button(chat_id)
            elif txt == "audio":
                tel_send_audio(chat_id, music_file_path)
            elif 'model' in txt:
                tel_send_document(chat_id)
            elif txt == "video":
                tel_send_video(chat_id)
            elif txt == "start" or txt == "Start" or txt == "/start":
                tel_send_inlinebutton(chat_id)
            elif 'secret' in txt:
                tel_send_inlineurl(chat_id)
            elif txt == "ic_A":
                tel_send_message(chat_id, "Send your url")
            elif txt == "ic_B":
                tel_send_message(chat_id, "Try me! You little Mozart!")
            elif ('http' in txt):
                yt = YouTube(txt)
                tel_send_message(chat_id, 'Downloading your file...')
                yt.streams.filter().get_by_resolution('360p').download(filename='temp.mp4')
                video = VideoFileClip('temp.mp4')
                video.audio.write_audiofile('temp.mp3')
                video.close()
                music_file_path = 'temp.mp3'
                tel_send_message(chat_id, 'Good taste')
                language = classify_mp3("", "temp.mp3")
                tel_send_message(chat_id, 'The lanuage is ' + language)
                tel_send_message(chat_id, 'detecting genre...')
                pred = classify(music_file_path, model)
                tel_send_message(chat_id, 'Your music looks like this to me')
                tel_send_image(chat_id, "image/temp/temp.jpg")
                tel_send_message(chat_id, 'I believe it belongs to ' + pred)
                if os.path.exists("temp.mp4"):
                    os.remove("temp.mp4")
                if os.path.exists("temp.mp3"):
                    os.remove("temp.mp3")
                if os.path.exists("temp.wav"):
                    os.remove("temp.wav")
                if os.path.exists("image/temp/temp.jpg"):
                    os.remove("image/temp/temp.jpg")
                tel_send_genre_image(chat_id, pred)
            else:
                tel_send_message(chat_id, "music!!!")
        except:
            print("fromindex-->")

        try:
            file_id, chat_id = tel_parse_get_message(msg)
            music_file_path = tel_upload_file(file_id)
            tel_send_message(chat_id, "What a song")
            language = classify_mp3("music/", os.listdir("music/")[0])
            tel_send_message(chat_id, 'The lanuage is ' + language)
            tel_send_message(chat_id, 'detecting genre...')
            pred = classify(music_file_path, model)
            tel_send_message(chat_id, 'Your music looks like this to me')
            tel_send_image(chat_id, 'image/temp/temp.jpg')
            tel_send_message(chat_id, 'I believe it belongs to ' + pred)
            # tel_send_image(chat_id)
            if os.path.exists(music_file_path):
                os.remove(music_file_path)
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
            if os.path.exists("image/temp/temp.jpg"):
                os.remove("image/temp/temp.jpg")
            tel_send_genre_image(chat_id, pred)
        except:
            print("No file from index-->")

        return Response('ok', status=200)
    else:
        return "<h1>Welcome!</h1>"


if __name__ == '__main__':
    app.run(threaded=True)
