import os

def split_into_gray_color(read_path,write_path):
    images = set()
    with open(read_path, 'r') as f:
        for line in f:
            images.add(line.strip())
    
    mid = len(images) // 2
    count = 0
    with open(write_path, 'w') as f:
        for img in images:
            assign_color(f, img,"gray" if count < mid else "color")
            count += 1
            
def assign_color(file_writer, image, color):
    file_writer.write(image + ',' + color + '\n')
    
test_images_path = r"C:\Users\daniy\OneDrive\Desktop\Thesis\data\splits\test.txt"
test_images_partition_path = r"C:\Users\daniy\OneDrive\Desktop\Thesis\data\splits\test_partitioned.txt"
train_images_path = r"C:\Users\daniy\OneDrive\Desktop\Thesis\data\splits\train.txt"
train_images_partition_path = r"C:\Users\daniy\OneDrive\Desktop\Thesis\data\splits\train_partitioned.txt"
split_into_gray_color(test_images_path,test_images_partition_path)
split_into_gray_color(train_images_path,train_images_partition_path)


