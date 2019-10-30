from Crypto.Cipher import AES
from Crypto import Random
import numpy
import cv2

class encryption():
    
    def __init__(self, key):
        self.key = Random.new().read(AES.block_size) #Creates key
        
    def encrypt(self, coordinates, image):
        IV = Random.new().read(AES.block_size) #Initialization vector
        mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        encryptor = AES.new(key, mode, IV) # Encryptor
        
        ROI_number = 0
        for c in coordinates:
            x,y,w,h = c
            ROI = image[y:y+h, x:x+w]
            img_bytes = numpy.ascontiguousarray(ROI, dtype=None)
        
            encData = encryptor.encrypt(img_bytes) #Encrypted data
        
            ROI_number += 1
            
        return encData

    def decrypt(self, coordinates):
        IV = Random.new().read(AES.block_size) #Initialization vector
        mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        decryptor = AES.new(key, mode, IV) # Decryptor
        
        for c in coordinates:
            encFile2 = open("Encrypted" + str(c) + ".enc", "rb") #Opens encrypted file created earlier
            encData2 = encFile2.read()
            encFile2.close()

            decData = decryptor.decrypt(encData2) #Decryption line
            
        return decData
