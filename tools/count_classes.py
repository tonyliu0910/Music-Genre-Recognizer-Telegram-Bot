import os

path = "../Images/"
class_list = os.listdir(path)
print(class_list)
class_dict = {}

for class_name in class_list:
    class_path = path + class_name + "/"
    image_list = os.listdir(class_path)
    class_dict[class_name] = len(image_list)

class_dict = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
for i in range(len(class_dict)):
    print(f"{class_dict[i][0]:<10}: {class_dict[i][1]}")
