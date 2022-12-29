import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "../Audio/"
image_path = "../Images/"

os.makedirs(image_path + "Pop", exist_ok=True)
os.makedirs(image_path + "Reggae", exist_ok=True)
os.makedirs(image_path + "Electronic", exist_ok=True)
os.makedirs(image_path + "Punk", exist_ok=True)
os.makedirs(image_path + "Jazz", exist_ok=True)
os.makedirs(image_path + "World", exist_ok=True)
os.makedirs(image_path + "Rap", exist_ok=True)
os.makedirs(image_path + "Folk", exist_ok=True)
os.makedirs(image_path + "Latin", exist_ok=True)
os.makedirs(image_path + "New Age", exist_ok=True)
os.makedirs(image_path + "Blues", exist_ok=True)
os.makedirs(image_path + "Metal", exist_ok=True)
os.makedirs(image_path + "Country", exist_ok=True)
os.makedirs(image_path + "Rock", exist_ok=True)
os.makedirs(image_path + "RnB", exist_ok=True)
# ['Pop', 'Reggae', 'Electronic', 'Punk', 'Jazz', 'World', 'Rap', 'Folk', 'Latin', 'New Age', 'Blues', 'Metal', 'Country', 'Rock', 'RnB']

file_list = os.listdir(audio_path)
for index in range(len(file_list)):
    if index == 462:
        continue
    filename = file_list[index]
    class_name = filename.split("_")[0]
    if os.path.isfile(image_path + class_name + "/" + filename[:-4] + ".jpg"):
        print("File already exists: " + filename)
        continue

    print(
        "Converting "
        + f"{index}"
        + " of "
        + f"{len(os.listdir(audio_path))}"
        + " files",
        end="",
    )
    print(" | filename: " + filename)

    plt.axis("off")
    plt.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])

    x, sr = librosa.load(audio_path + filename, sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")

    plt.savefig(
        image_path + class_name + "/" + filename[:-4] + ".jpg",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close()
