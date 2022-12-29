import torchaudio
import os
import librosa
import matplotlib.pyplot as plt

data_path = "zh"
save_path = "zh_spectrogram"

os.makedirs(save_path, exist_ok=True)
files = os.listdir(data_path)
print(len(files))
transform = torchaudio.transforms.MelSpectrogram(n_mels=64)

length = len(files)

for index, f in enumerate(files):
    print(f)
    if not f.endswith(".mp3"):
        continue
    waveform, sampleRate = torchaudio.load(os.path.join(data_path, f))
    waveform = torchaudio.functional.resample(waveform, sampleRate, 16000)
    spectrogram = transform(waveform)[0]
    plt.imshow(librosa.power_to_db(spectrogram), origin="lower", aspect="auto")
    plt.axis("off")
    plt.savefig(os.path.join(save_path, f.split(".")[0] + ".png"))

    print(
        f"Processing {index:>7} / {length:>7}|"
        + "-" * int(50 * index / length)
        + ">"
        + " " * (50 - int(50 * index / length))
        + "|"
        + f"{index / length:2f}%",
        end="\r",
    )

#  haven't tested this yet, but it should work
