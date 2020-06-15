from db_connection import insert_image, insert_bbox
import os
import mysql.connector
import sys
import config

class ImageDB:

    # def __init__(self):

    #     self.mysql = SQLConn()
    

    # def init_tables(self):
        
    #     #setup tables in sql server
    #     try:
    #         self.mysql.CreateImagetable()
    #         self.mysql.CreateBBoxtable()
    #     except:
    #         pass

    def insert_image(self, image_name, image_date, image_time, init_vector):
        """Insert image metadata into sql database

        Args:
            image_name (string): name of image
            image_date (date obj): date image was taken
            image_time (time obj): time image was taken (24hr format?)
            init_vector (string): decryption key for image
        """
        try:
            image = [image_name, image_date, image_time, init_vector]

            self.mysql.insertImage(tuple(image))
            self.mysql.mydb.commit()
        
        except mysql.connector.Error as e:
            print('Error inserting image information: ' + str(e))
            self.mysql.mydb.rollback()
            self.mysql.mydb.commit()
            pass
        except Exception as e2:
            self.mysql.mydb.rollback()
            self.mysql.mydb.commit()
            print(e2)
            pass


    def insert_bbox(self, x_min, y_min, x_max, y_max, confidence, goggles, image_name):
        """Insert bounding box data into sql database

        Args:
            x_min (float): x coordinate of bottom left corner of bbox
            y_min (float): y coordinate of bottom left corner of bbox
            x_max (float): x coordinate of top right corner of bbox
            y_max (float): y coordinate of top right corner of bbox
            confidence (float): confidence of bbox classification
            goggles (bool): presence of no goggles
            image_name (string): name of image
        """
        try:    
            bbox = [x_min, y_min, x_max, y_max, confidence, goggles, image_name]

            self.mysql.insertImage(tuple(bbox))
            self.mysql.mydb.commit()
        
        except mysql.connector.Error as e:
            print('Error inserting bbox information: ' + str(e))
            self.mysql.mydb.rollback()
            self.mysql.mydb.commit()
            pass
        except Exception as e2:
            self.mysql.mydb.rollback()
            self.mysql.mydb.commit()
            print(e2)
            pass

        
    #def insert_data(self):

    
    