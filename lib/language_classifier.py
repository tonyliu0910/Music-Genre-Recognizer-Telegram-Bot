import torch
import torchvision
import torchaudio
import librosa
import matplotlib.pyplot as plt
import cv2
import os

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and weights
model = torchvision.models.resnet18(weights=None).to(device)
model.fc = torch.nn.Linear(512, 7)
model.load_state_dict(torch.load("weights/ResNet-18.ckpt"))
model.to(device)
model.eval()

transform = torchaudio.transforms.MelSpectrogram(n_mels=64)

inference_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def classify_img(img):
    with torch.no_grad():
        img = inference_transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        pred = model(img)
        pred = torch.argmax(pred, dim=-1)
    if pred == 0:
        return "Chinese"
    elif pred == 1:
        return "English"
    elif pred == 2:
        return "Japanese"
    elif pred == 3:
        return "French"
    elif pred == 4:
        return "Spanish"
    elif pred == 5:
        return "Thai"
    elif pred == 6:
        return "Russian"


def classify_mp3(mp3_floder, mp3_path):
    waveform, sampleRate = torchaudio.load(
        os.path.join(mp3_floder, mp3_path), format="mp3"
    )
    waveform = torchaudio.functional.resample(waveform, sampleRate, 16000)
    spectrogram = transform(waveform)[0]
    plt.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto")
    plt.axis("off")
    plt.savefig("test.png")
    plt.close()

    img = cv2.imread("test.png")
    return classify_img(img)


if __name__ == "__main__":
    print(classify_mp3("", "test.mp3"))
