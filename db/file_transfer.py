import paramiko
import os
import sys
import config

class FTPConn:
    
    def __init__(self):
       
        try:
            self.transport = paramiko.Transport(('', 22))
            self.transport.connect(username='', password='')
            self.sftp = paramiko.SFTPClient.from_transport(self.transport)
            print('Sucessfully connected to host machine')

        except Exception as e:
            print(e)
            print('Error connecting to the host machine')

    def transfer(self, input_dir, output_dir):

        if not input_dir.endswith('/'):
            input_dir = input_dir + '/'

        if not output_dir.endswith('/'):
            output_dir = output_dir + '/'

        try:
            for image in os.listdir(input_dir):
                if image.endswith('.py'):
                    self.sftp.put(input_dir + image, output_dir + image)
        
        except Exception as e:
            print(e)
    
    def disconnect(self):

        self.sftp.close()
        self.transport.close()
