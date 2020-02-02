#coding=utf-8

import argparse
import cv2
import h5py
import os
import pathlib
import sys
import json
import shutil
import time
import datetime
from multiprocessing import Pool
from functools import partial

import preprocess as prep

INPUT_SIZE = (2048, 128, 1)
PREDICTION_LINE_CODE = 'P'

def progress_bar (flag, total):
    bar = '--------------------------------------------------'
    percent = flag * 100//total 
    if percent % 2 == 0:
        bar = bar.replace('-','#',percent//2)
        return bar , percent
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--start", action="store_true", default=False)
    args = parser.parse_args()
    if args.start:
        print("Enter the data type (train, val or test):    ")
        data_type = str(input())
        output_path = os.path.join('..', 'data', data_type)
        prediction_file = os.path.join('..', 'evaluation', 'predictions_{}.txt'.format(data_type))

        print(f"Enter the path to {data_type} data:    ")
        raw_path = str(input())
        assert os.path.exists(raw_path)
        
        os.makedirs(os.path.join(output_path), exist_ok=True)
        image_paths = [str(item) for item in pathlib.Path(raw_path).glob('*') if item.is_file() and not str(item.name).endswith('json')]
        assert len(image_paths) > 0
        label_path = [str(item) for item in pathlib.Path(raw_path).glob('*') if item.is_file() and str(item.name).endswith('json')]
        assert len(label_path) == 1
        label_file_name = [str(item.name) for item in pathlib.Path(raw_path).glob('*') if item.is_file() and str(item.name).endswith('json')][0]
        shutil.copy(label_path[0],os.path.join(output_path,label_file_name))
        flag = 0
        total = len(image_paths)
        bar = '--------------------------------------------------'
        percent = 1
        epoch = 200
        for path in image_paths:
            print('Transforming ==>')
            print(f"{bar} ::: {percent} %")
            print(f"About {round(epoch) * (total-flag)} s to finish ")
            start_time = time.time()
            prep.preprosess_raw(img_path=path, input_size=INPUT_SIZE, save_type = data_type)
            epoch = time.time() - start_time
            print(f"About {round(epoch) * (total-flag)} s to finish ")
            flag    += 1
            if isinstance(progress_bar (flag, total), (tuple,str)):
                bar, percent = progress_bar (flag, total)
            os.system( 'cls' )           
            print('Transforming =====>')
            print(f"{bar} ::: {percent} %")
            print(f"About {round(epoch) * (total-flag)} s to finish ")
            time.sleep(0.2)
            os.system( 'cls' )
            print('Transforming =========>')
            print(f"{bar} ::: {percent} %")
            print(f"About {round(epoch) * (total-flag)} s to finish ")
            time.sleep(0.2)
            os.system( 'cls' )
            print('Transforming ===============>')
            print(f"{bar} ::: {percent} %")
            print(f"About {round(epoch) * (total-flag)} s to finish ")
            time.sleep(0.2)
            os.system( 'cls' )
        print('-------------------------------------------------------------------------------')
        print (f"Transformed {total} data, check {output_path} for new {data_type} set")
        print('-------------------------------------------------------------------------------')

    elif args.sample:
        print("Enter the data type (train, val or test):    ")
        data_type = str(input())
        output_path = os.path.join('..', 'data', data_type)
        prediction_file = os.path.join('..', 'evaluation', 'predictions_{}.txt'.format(data_type))
        
        print("Enter the SAMPLE_SIZE:    ")
        SAMPLE_SIZE = int(input())


        labels = json.load(open(os.path.join('..', 'data',data_type , '{}.json'.format(data_type)), encoding = 'utf-8'))
        image_paths = [str(item) for item in pathlib.Path(os.path.join(output_path)).glob('*') if item.is_file()][:SAMPLE_SIZE]
        if os.path.isfile(prediction_file):
            with open(prediction_file, 'r') as f:
                predictions = [line[len(PREDICTION_LINE_CODE):] for line in f if line.startswith(PREDICTION_LINE_CODE)]
        else:
            predictions = [''] * SAMPLE_SIZE
        for i in range(SAMPLE_SIZE):
            image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
            file_name = pathlib.Path(image_paths[i]).name
            print("Image shape:\t{}".format(image.shape))
            print("Label:\t{}".format(labels[file_name]))
            print("Predict:\t{}\n".format(predictions[i]))

            cv2.imshow("img", prep.adjust_to_see(image))
            cv2.waitKey(0)
    