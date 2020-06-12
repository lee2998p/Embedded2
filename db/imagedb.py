from db_connection import SQLConn
import os
import mysql.connector
import sys
import config
import time                                                                                 #time_measurement

class ImageDB:

    def __init__(self):

        self.mysql = SQLConn()
    

    def init_tables(self):
        
        try:
            self.mysql.CreateImagetable()
            self.mysql.CreateBBoxtable()
        except:
            pass

    #def check_header(self):

    def insert_image(self, image_name, image_date, image_time, init_vector):
    
        start_time = time.time()                                         #time_measurement
        try:
            name = os.path.splitext(image_name)[0]
            
            image = [image_name, image_date, image_time, init_vector]

            self.mysql.insertImage(tuple(image))
            self.mysql.mydb.commit()
            
            print("The insertion of the image took %s seconds." (time.time() - start_time))           #time_measurement
        
        except mysql.connector.Error as e:
            print('Error inserting image information: ' + str(e))
            self.mysql.mydb.rollback()
            self.mysql.mydb.commit()
            pass
        except Exception as e2:
            print(e2)
            pass


    def insert_bbox(self, x_min, y_min, x_max, y_max, confidence, goggles, image_name):
        start_time = time.time()                                                                  #time_measurement
        try:
            name = os.path.splitext(image_name)[0]
            
            bbox = [x_min, y_min, x_max, y_max, confidence, goggles, name]

            self.mysql.insertImage(tuple(bbox))
            self.mysql.mydb.commit()
            print("The insertion of the bounding box took %s seconds." (time.time() - start_time))         #time_measurement
        
        except mysql.connector.Error as e:
            print('Error inserting bbox information: ' + str(e))
            self.mysql.mydb.rollback()
            self.mysql.mydb.commit()
            pass
        except Exception as e2:
            print(e2)
            pass

        
    #def insert_data(self):

    
    
