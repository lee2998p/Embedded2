#!/usr/bin/env python3
from Crypto.Cipher import AES
import cv2
import sys
import os
import numpy as np


class decryption():

    def __init__(self, key):
        self.key = key  # Creates key using KDF scheme

    def decrypt(self, coordinates, image, IV):
        # IV = Random.new().read(AES.block_size) #Initialization vector
        mode = AES.MODE_CFB  # Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        decryptor = AES.new(self.key, mode, IV)  # Decryptor

        for c in coordinates:
            x1, y1, x2, y2 = c
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            ROI = image[y1:y2, x1:x2]
            img_bytes = np.ascontiguousarray(ROI, dtype=None)
            print (img_bytes.shape)

            decData = decryptor.decrypt(img_bytes)  # Encrypted data
            data = np.frombuffer(decData, dtype=np.uint8)
            data = np.reshape(data, (y2 - y1, x2 - x1, 3))
            image[y1:y2, x1:x2] = data

        return image

    def get_record(self, image_timestamp):
        metadata_filename = image_timestamp + '.txt'
        metadata_image = image_timestamp + '.png'

        encrypted_image = cv2.imread(metadata_image)

        f = open(metadata_filename, "r")
        contents = f.readlines()
        f.close()

        contents = contents[0].split('|')
        IV = int(contents[2])
        IV_in_bytes = (IV).to_bytes((IV.bit_length() + 7) // 8, byteorder='big')

        face_coordinates = contents[1].replace('[', '').replace(']', '').split(',')
        number_of_rows = int(len(face_coordinates) / 4)
        np_coordinates = np.asarray(face_coordinates, dtype=np.float32)
        np_coordinates = np.reshape(np_coordinates, (number_of_rows, 4))
        face_coordinates = np_coordinates.tolist()

        return encrypted_image, IV_in_bytes, face_coordinates



if __name__ == "__main__":
    face_coordinates = [[336, 42, 416, 180], [560, 61, 646, 212], [243, 84, 298, 151]]

    key_input = int(input('Enter key: '))
    key_in_bytes = (key_input).to_bytes((key_input.bit_length() + 7) // 8, byteorder='big')


    image_decryption = decryption(key_in_bytes)


    #TODO: check_key()

    image_timestamp_input = input('Enter Image Timestamp (24h time) in this format - (yyyy-mm-dd-hh-mm-ss): ')

    image, IV, face_coordinates = image_decryption.get_record(image_timestamp_input)

    decrypted_image = image_decryption.decrypt(face_coordinates, image, IV)

    filename = image_timestamp_input + '_decrypted' + '.jpg'
    visualize_input = input('Visualize Image? (Y / N) ')
    if visualize_input == 'Y':
        cv2.imshow('frame', decrypted_image)
        cv2.imwrite(filename, decrypted_image)
    else:
        cv2.imwrite(filename, decrypted_image)




    '''
    key = 18315565048028305729012006914309239581
    image = 2020-03-26-21-12-09
    '''
