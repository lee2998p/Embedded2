from ftplib import FTP
import os
import mysql.connector
import sys
import config

class FTPConn:
    
    def __init__(self):
       
        try:
            server_connect = ftplib.FTP('ee220clnx1.ecn.purdue.edu',
                                        '', 
                                        '')

            print('Sucessfully connected to host machine')

        except:
            print('Error with connecting to the host machine')
    
    def transfer(self, input_path, output_path, batch_size):                      
        for x in range(batch_size):         #for/while loop for sending files   condition for the loop
            try:
                file = open(image, 'rb')
                server_connect.storbinary('STOR' + str(image), file)     #will have to make sure that the image is stored to the correct place in the helps machine
                                                                #Can I use a place holder or an index?
                file.close()
                print('File was transferred correctly')
            except:
                print('Error with transferring image')
                pass
                
     def disconnect(self): 
        try:
            server_connect.quit()
        except:
            print('Error with disconnecting from the helps machine')
