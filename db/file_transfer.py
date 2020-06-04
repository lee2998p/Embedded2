from ftplib import FTP
import os
import mysql.connector
import sys
import config

class FTP_Conn:
    
    def __init__(self):
       
        try:
            server_connect = ftplib.FTP(host, username, password, account)
        
        except:
            print('Error with connecting to the helps machine')
       
        pass:
            exit()
    
    def transfer:                       #input: location in jetson nano and location in the helps machine
        for x in range(10):         #for/while loop for sending files   condition for the loop
            try:
                file = open(image, 'rb')
                server_connect.storbinary(STOR image, file)     #will have to make sure that the image is stored to the correct place in the helps machine
                                                                #Can I use a place holder or an index?
                file.close()
                print('File was transferred correctly')
            except:
                print('Error with transferring image')
                pass
                
     def disconnect: 
        try:
            server_connect.quit()
        except:
            print('Error with disconnecting from the helps machine')
