import os

audio_path = "../Audio/"
image_path = "../Images/"

file_list = os.listdir(audio_path)
for index, file in enumerate(file_list):
    if os.path.getsize(audio_path + file) < 2**11 and os.path.isfile(
        image_path + file.split("_")[0] + "/" + file[:-4] + ".jpg"
    ):
        print(
            "Removing " + file + " with size " + str(os.path.getsize(audio_path + file))
        )
        os.remove(image_path + file.split("_")[0] + "/" + file[:-4] + ".jpg")
