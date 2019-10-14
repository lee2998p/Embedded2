from Crypto.Cipher import AES
from Crypto import Random
import matplotlib.pyplot as plt
from PIL import Image

def encrypt(encryptor):
    img_bytes = open("TestImage.jpg", "rb").read() #Reads image bytes
    print(len(img_bytes))
    encData = encryptor.encrypt(img_bytes) #Encryption line
    
    encFile = open("Encrypted.enc", "wb") #creates encrypted file in directory
    encFile.write(encData)
    encFile.close()

def decrypt(decryptor):

    encFile2 = open("Encrypted.enc", "rb") #Opens encrypted file created earlier
    encData2 = encFile2.read()
    encFile2.close()

    decData = decryptor.decrypt(encData2) #Decryption line

    output_file = open("output.jpeg", "wb") #Creates decrypted file in directory
    output_file.write(decData)
    output_file.close()

if __name__=="__main__":

    key = Random.new().read(AES.block_size) #Creates key
    IV = Random.new().read(AES.block_size) #Initialization vector
    mode = AES.MODE_CFB #Sets encryption mode to CFB mode; CFB is great in avoiding the hassle of padding
    
    encryptor = AES.new(key, mode, IV) # Encryptor
    decryptor = AES.new(key, mode, IV) # Decryptor

    encrypt(encryptor) #Calls encryption function
    decrypt(decryptor) #Calls decryption function