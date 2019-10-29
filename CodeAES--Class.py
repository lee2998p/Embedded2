from Crypto.Cipher import AES
from Crypto import Random
#from matplotlib.pyplot import imread
#from PIL import Image
import numpy
import cv2

class encryption():
    
    def __init__(self, key):
        self.key = Random.new().read(AES.block_size) #Creates key
        
    def encrypt(key, coordinates, image):
        IV = Random.new().read(AES.block_size) #Initialization vector
        mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        encryptor = AES.new(key, mode, IV) # Encryptor
        
        ROI_number = 0
        for c in coordinates:
            x,y,w,h = c
            ROI = image[y:y+h, x:x+w]
            img_bytes = numpy.ascontiguousarray(ROI, dtype=None)
        
            #cv2.imshow("Image", ROI) #Just shows ROI
            #cv2.waitKey(0) #For showing ROI
        
            encData = encryptor.encrypt(img_bytes) #Encrypted data
        
            #encFile = open("Encrypted" + str(c) + ".enc", "wb") #creates encrypted file in directory
            #encFile.write(encData)
            #encFile.close()
            return encData
        
            ROI_number += 1

    def decrypt(key, coordinates):
        IV = Random.new().read(AES.block_size) #Initialization vector
        mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
        decryptor = AES.new(key, mode, IV) # Decryptor
        
        for c in coordinates:
            encFile2 = open("Encrypted" + str(c) + ".enc", "rb") #Opens encrypted file created earlier
            encData2 = encFile2.read()
            encFile2.close()

            decData = decryptor.decrypt(encData2) #Decryption line

            #output_file = open("Decrypted_ROI" + str(c) + ".jpg", "wb") #Creates decrypted file in directory
            #output_file.write(decData)
            #output_file.close()
            
            return decData
