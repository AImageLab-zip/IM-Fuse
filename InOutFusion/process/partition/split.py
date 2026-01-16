import os
import random

# 定义文件路径和目标文件名
input_path = r"/Code/SFusion-main/dataset/MICCAI_BraTS_2018_Data_Training/HGG"  # 
output_train = "2018_0-train.txt"
output_test = "2018_0-test.txt"
output_val = "2018_0-val.txt"

# 获取文件列表
file_list = os.listdir(input_path)
file_list = [os.path.join(input_path,filename) for filename in file_list]
print(file_list)
# 打乱文件顺序
random.shuffle(file_list)

# 按7:2:1分割
train_split = int(len(file_list) * 0.7)
test_split = int(len(file_list) * 0.9)

train_files = file_list[:train_split]
test_files = file_list[train_split:test_split]
val_files = file_list[test_split:]

print(len(train_files))
# 保存文件路径到对应的txt文件
# 保存文件路径到对应的txt文件
def save_to_file(filename, file_list):
    with open(filename, 'w') as file:
        for path in file_list:
            file.write(path + '\n')

save_to_file(output_train, train_files)
save_to_file(output_test, test_files)
save_to_file(output_val, val_files)

print("路径分割完成并保存至对应的txt文件中：")
print(f"训练集文件保存到：{output_train}")
print(f"测试集文件保存到：{output_test}")
print(f"验证集文件保存到：{output_val}")# 保存文件路径到对应的txt文件
def save_to_file(filename, file_list):
    with open(filename, 'w') as file:
        for path in file_list:
            file.write(path + '\n')

save_to_file(output_train, train_files)
save_to_file(output_test, test_files)
save_to_file(output_val, val_files)

print("路径分割完成并保存至对应的txt文件中：")
print(f"训练集文件保存到：{output_train}")
print(f"测试集文件保存到：{output_test}")
print(f"验证集文件保存到：{output_val}")

print("路径分割完成并保存至对应txt文件中！")
