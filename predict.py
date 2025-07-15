import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np

# 你的类别列表（建议和训练时保持一致）
all_classes = sorted([
    d for d in os.listdir('AffectNet/train') if os.path.isdir(os.path.join('AffectNet/train', d))
])
class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# 加载模型
model = tf.keras.models.load_model('model_FERP.keras', compile=False)

# 图像预处理
IMG_SIZE = 120

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = tf.expand_dims(x, axis=0)
    # 这里你训练时用的是 mobilenet 的 preprocess_input
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    return x

dataset_root = 'AffectNet'
splits = ['train', 'val', 'test']

results = []
id_counter = 1

for split in splits:
    split_dir = os.path.join(dataset_root, split)
    for true_class in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, true_class)
        if not os.path.isdir(class_dir):
            continue
        true_label = class_to_idx.get(true_class, -1)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                x = preprocess_img(img_path)
                preds = model.predict(x, verbose=0)
                pred_label = np.argmax(preds, axis=1)[0]
                pred_class = idx_to_class[pred_label]

                results.append([
                    id_counter,
                    os.path.join(split, true_class, img_name),
                    true_label,
                    true_class,
                    pred_label,
                    pred_class,
                ])
                id_counter += 1
            except Exception as e:
                print(f"[错误] 处理图像失败: {img_path}, 原因: {e}")

# 保存到 Excel
df = pd.DataFrame(results, columns=["id", "image_name", "true_label", "true_class", "pred_label", "pred_class"])
df.to_excel("predict_AffectNet.xlsx", index=False)
print("预测完成，结果已保存到 predict_AffectNet.xlsx")
