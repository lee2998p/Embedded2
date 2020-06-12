import paramiko
import os
import sys
import config
import time                                                                                         #time_measurement

class FTPConn:
    
    def __init__(self):
       #setup connection with host machine
        try:
            self.transport = paramiko.Transport((config.FTPHOST, 22))
            self.transport.connect(username=config.FTPUSER, password=config.FTPPASS)
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)
            print('Sucessfully connected to host machine')

        except Exception as e:
            print(e)
            print('Error connecting to the host machine')

    def transfer(self, input_dir, output_dir):
        start_time = time.time()                                                                    #time_measurement
        #make sure directory paths are in correct format
        if not input_dir.endswith('/'):
            input_dir = input_dir + '/'
        if not output_dir.endswith('/'):
            output_dir = output_dir + '/'

        #transfer files from input directory to output directory
        try:
            for image in os.listdir(input_dir):
                #only transfer files that end with certain type(.py for testing purposes)
                if image.endswith('.py'):
                    self.sftp.put(input_dir + image, output_dir + image)
                    #os.remove(input_dir + image) #remove the image from client side after successful transfer
            print("The transfer took %s seconds." % (time.time() - start_time))                     #time_measurement
            
        
        except Exception as e:
            print(e)
    
    
    def disconnect(self):
        
        #disconnect sftp session
        self.sftp.close()
        self.transport.close()
