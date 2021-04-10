import os
import random

#data_dir是image文件夹
def get_image(data_dir):
    for root, dirs, _ in os.walk(data_dir):
        for sub_dir in dirs:
            img_names = os.listdir(os.path.join(root, sub_dir))
            train_imgs = img_names[:int(len(img_names) * scale_factor)]
            #test_imgs = img_names[int(len(img_names) * scale_factor):]
            rest_imgs = img_names[int(len(img_names) * scale_factor):]
            val_imgs = rest_imgs[int(int(len(rest_imgs)) / 2):]
            test_imgs = rest_imgs[:int(int(len(rest_imgs)) / 2)]

            #print(train_imgs)
            if sub_dir == 'RGB':
                f = open('./data/RGBT/train_RGB.txt', 'w')
                for i in range(len(train_imgs)):
                    img_name = train_imgs[i]
                    #print('RGB:', img_name)
                    f.write(img_name + '\n')

                p = open('./data/RGBT/test_RGB.txt', 'w')
                for i in range(len(test_imgs)):
                    img_name = test_imgs[i]
                    p.write(img_name + '\n')

                q = open('./data/RGBT/val_RGB.txt', 'w')
                for i in range(len(val_imgs)):
                    img_name = val_imgs[i]
                    q.write(img_name + '\n')
            else:
                f = open('./data/RGBT/train_T.txt', 'w')
                for i in range(len(train_imgs)):
                    img_name = train_imgs[i]
                    #print('T:', img_name)
                    f.write(img_name + '\n')

                p = open('./data/RGBT/test_T.txt', 'w')
                for i in range(len(test_imgs)):
                    img_name = test_imgs[i]
                    p.write(img_name + '\n')

            #valid_name = random.shuffle(img_names)
            #valid_imgs = valid_name[:200]


if __name__ == '__main__':
    data_dir = 'D:\Documents\DSS-pytorch-master\data\RGBT\image'
    scale_factor = 0.8
    get_image(data_dir)
