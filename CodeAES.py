from Crypto.Cipher import AES
from Crypto import Random
#from matplotlib.pyplot import imread
from PIL import Image
import numpy
import cv2

def encrypt(encryptor, coordinates):
    ###################################################################
    image = cv2.imread('TestImage.jpg')

    ROI_number = 0
    for c in coordinates:
        cv2.rectangle(image, (400, 400), (600, 200), (0, 0, 255), 1)
        x,y,w,h = c
        ROI = image[y:y+h, x:x+w]
        img_bytes = numpy.ascontiguousarray(ROI, dtype=None)
        
        encData = encryptor.encrypt(img_bytes) #Encrypted data
        
        encFile = open("Encrypted.enc", "wb") #creates encrypted file in directory
        encFile.write(encData)
        encFile.close()
        
        ROI_number += 1
    #################################################################

def decrypt(decryptor):

    encFile2 = open("Encrypted.enc", "rb") #Opens encrypted file created earlier
    encData2 = encFile2.read()
    encFile2.close()

    decData = decryptor.decrypt(encData2) #Decryption line

    output_file = open("NewOutput.jpeg", "wb") #Creates decrypted file in directory
    output_file.write(decData)
    output_file.close()

if __name__=="__main__":    
    
    key = Random.new().read(AES.block_size) #Creates key
    IV = Random.new().read(AES.block_size) #Initialization vector
    mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
    
    coordinates = [(200, 300, 400, 250), (500, 800, 830, 612)]
    
    encryptor = AES.new(key, mode, IV) # Encryptor
    decryptor = AES.new(key, mode, IV) # Decryptor

    encrypt(encryptor, coordinates) #Calls encryption function
    #decrypt(decryptor) #Calls decryption function
