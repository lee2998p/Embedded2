from ftplib import FTP
import os
import mysql.connector
import sys
import config

class FTP_Conn:
    
    def __init__(self):
       
        server_connect = ftplib.FTP(host, username, password, account)
        for x in range(10):         #for/while loop for sending files   condition for the loop
            file = open(image, 'rb')
            server_connect.storbinary(STOR image, file)     #will have to make sure that the image is stored to the correct place in the helps machine
                                                            #Can I use a place holder or an index?
            file.close()
        server_connect.quit()
