#encoding='utf-8'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

import h5py
import numpy as np
import unicodedata
import json
import os
import pathlib
import cv2
import datetime
import argparse
import itertools

import preprocess as prep
import model as mb
import evaluation

char = "¶ #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstwuvxyzÂÊÔàáâãèéêẹìíòóôõùúýăĐđĩũƠơưạảấầẩẫậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE
vocab_size = len(char)
MAX_LABEL_LENGTH = 128
INPUT_SIZE = (2048, 128, 1)

# convert the words to array of indexs (and prevert) based on the char_list
def text_to_labels(text):
    return np.asarray(list(map(lambda x: char.index(x), text)), dtype=np.uint8)

def labels_to_text(labels):
    return ''.join(list(map(lambda x: char[x] if x < len(char) else "", labels)))
def preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)/255
    image = tf.image.per_image_standardization(image)
    return image
def load_and_preprocess_image(path, label):
    return preprocess(path), label

def augmentation(imgs,
                 rotation_range=0,
                 scale_range=0,
                 height_shift_range=0,
                 width_shift_range=0,
                 dilate_range=1,
                 erode_range=1):
    """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

    imgs = imgs.numpy().astype(np.float32)
    h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]
    imgs = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    imgs = cv2.erode(imgs[i], erode_kernel, iterations=1)
    imgs = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs

@tf.function
def augmentation_tf(images, label):
    image = tf.py_function(func = augmentation,  inp = [images],Tout=tf.float32)
    return images, label

def create_dataset(type_, cache=False, training = True, augment = False):
    data_folder = os.path.join('..', 'data', type_)
    ds = tf.data.Dataset.list_files(os.path.join(data_folder,'*'))
    false_path1 = os.path.join(data_folder,'.ipynb_checkpoints')
    false_path2 = os.path.join(data_folder,type_+'.json')
    all_paths = [str(item) for item in pathlib.Path(data_folder).glob('*') if str(item) !=  false_path2 and str(item) !=  false_path1]
    label_dic = json.load(open(os.path.join('..', 'data', type_ , '{}.json'.format(type_)),encoding = 'utf-8'))
    labels = [label_dic[pathlib.Path(path).name] for path in all_paths] 
    labels = [prep.text_standardize(label) for label in labels]
    label_int = [text_to_labels(label) for label in labels]
    label_int = pad_sequences(label_int, maxlen = MAX_LABEL_LENGTH, padding = 'post')
    n_samples = len(label_int)
    steps_per_epoch = tf.math.ceil(n_samples/BATCH_SIZE)
    ds = tf.data.Dataset.from_tensor_slices((all_paths, label_int))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.cache()
        ds = ds.shuffle(buffer_size = SHUFFLE_BUFFER_SIZE)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        ds.map(augmentation_tf, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(BATCH_SIZE)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds, steps_per_epoch, labels

def decode_batch(out):
    result = []
    for j in range(out[0].shape[0]):
        out_best = list(np.argmax(out[0][j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        result.append(outstr)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    # checkpoint = './checkpoint_weights.hdf5'
    if args.train:
        print ('Chose your model (m1 or m2)')
        md = str(input())      

        train_ds, num_steps_train, _ = create_dataset('train', cache=True)
        test_ds, num_steps_val, _    = create_dataset('val', training=False)
        if md == 'm1':
            model = mb.build_model1(input_size=INPUT_SIZE, d_model=vocab_size+1, learning_rate=0.001)
        else:
            model = mb.build_model2(input_size=INPUT_SIZE, d_model=vocab_size+1, learning_rate = 0.001)
#         model.load_weights(checkpoint)
        model.summary()
        batch_stats_callback = mb.CollectBatchStats()
        callbacks = mb.Callbacks
        start_time = datetime.datetime.now()

        h = model.fit(train_ds,
                    steps_per_epoch = num_steps_train,
                    epochs=100,
                    validation_data = test_ds,
                    validation_steps = num_steps_val,
                    callbacks=callbacks)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))

        t_corpus = "\n".join([
            "Batch:                   {}\n".format(BATCH_SIZE),
            "Time per epoch:          {}".format(time_epoch),
            "Total epochs:            {}".format(len(loss)),
            "Best epoch               {}\n".format(min_val_loss_i + 1),
            "Training loss:           {}".format(loss[min_val_loss_i]),
            "Validation loss:         {}".format(min_val_loss),
        ])

        with open(os.path.join("train.txt"), "w") as f:
            f.write(t_corpus)
            print(t_corpus)
    # Testing
    elif args.test:
        checkpoint = '../data/weight/checkpoint_weights(2048_m1).hdf5' 
        print('Enter path to test data')
        test_path = str(input())
        assert os.path.isfile(checkpoint) and os.path.exists(test_path)
        type_ = pathlib.Path(test_path).name
        ds, num_steps, labels = create_dataset(type_, training=False)
        model = mb.build_model1(input_size=INPUT_SIZE, d_model=vocab_size+1)
        model.load_weights(checkpoint)
        model.summary()

        start_time = datetime.datetime.now()

        predictions = model.predict(ds, steps=num_steps)

        # CTC decode
        ctc_decode = True
        if ctc_decode:
            predicts, probabilities = [], []
            x_test = np.array(predictions)
            x_test_len = [MAX_LABEL_LENGTH for _ in range(len(x_test))]

            decode, log = K.ctc_decode(x_test,
                                    x_test_len,
                                    greedy=True,
                                    beam_width=10,
                                    top_paths=1)

            probabilities = [np.exp(x) for x in log]
            predicts = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts = np.swapaxes(predicts, 0, 1)
            predicts = [labels_to_text(label[0]) for label in predicts]
        else:
            predicts = decode_batch(predictions)

        total_time = datetime.datetime.now() - start_time
        print(predicts[:10])
        print(labels[:10])
#         predicts = [x.replace(PAD_TK, "") for x in predicts]
        prediction_file = os.path.join('.', 'predictions_{}.txt'.format(type_))

        with open(prediction_file, "w", encoding='utf-8') as f:
            for pd, gt in zip(predicts, labels):
                f.write("Y {}\nP {}\n".format(gt, pd))

        evaluate = evaluation.ocr_metrics(predicts=predicts,
                                          ground_truth=labels,
                                          norm_accentuation=False,
                                          norm_punctuation=False)

        e_corpus = "\n".join([
            "Total test images:    {}".format(len(labels)),
            "Total time:           {}".format(total_time),
            "Metrics:",
            "Character Error Rate: {}".format(evaluate[0]),
            "Word Error Rate:      {}".format(evaluate[1]),
            "Sequence Error Rate:  {}".format(evaluate[2]),
        ])

        with open("evaluate.txt", "w") as lg:
            lg.write(e_corpus)
            print(e_corpus)