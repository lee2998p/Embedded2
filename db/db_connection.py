import mysql.connector
#import _mysql_connector
from mysql.connector import errorcode

import sys
import config

class SQLConn:

    def __init__(self):
        """setup connection to MySQL database
        """

        #Specify database
        mydatabase = config.KEYSPACE

        try:
            self.mydb = mysql.connector.connect(
                host=config.SQL_HOST,
                user=config.USER_NAME,
                password=config.PASSWORD,
                database= mydatabase
            )
            print('Connected to mysql databse' + mydatabase)

        except mysql.connector.Error as err:
            print(str(err))
            sys.exit()
        except _mysql_connector.MySQLInterfaceError as e:
            print(str(e))
            sys.exit()
        except Exception as e:
            print(str(e))
            sys.exit()

        self.mycursor = self.mydb.cursor(buffered=True)


    def dropImageTable(self):
        """Delete image table
        """
        self.mycursor.excute('DROP TABLE IF EXISTS IMAGE')
        print('IMAGE table dropped')
    

    def dropBBoxTable(self):
        """Delete bounding box table
        """
        self.mycursor.execute('DROP TABLE IF EXISTS BBOX')
        print('BBOX table dropped')
    

    def CreateImageTable(self):
        """Create image table
        """
        try:
            self.mycursor.execute('SELECT 1 FROM IMAGE LIMIT 1')
            print('IMAGE table exists')
        
        except:
            self.mycursor.execute('CREATE TABLE IMAGE(Image_Name VARCHAR(25) NOT NULL,\
                                    Image_Date DATE NOT NULL,\
                                    Image_Time TIME NOT NULL,\
                                    Init_Vector VARCHAR(25) NOT NULL,\
                                        PRIMARY KEY(Image_Name))')
            print('IMAGE table created')


    def CreateBBoxTable(self):
        """Create bounding box table
        """
        try:
            self.mycursor.execute('SELECT 1 FROM BBOX LIMIT 1')
            print('BBOX table exists')
        
        except:
            self.mycursor.execute('CREATE TABLE BBOX(X_Min VARCHAR(10) NOT NULL,\
                                    Y_Min VARCHAR(10) NOT NULL,\
                                    X_Max VARCHAR(10) NOT NULL,\
                                    Y_Max VARCHAR(10) NOT NULL,\
                                    Confidence VARCHAR(10) NOT NULL,\
                                    Goggles VARCHAR(10) NOT NULL,\
                                    Image_Name VARCHAR(25) NOT NULL,\
                                        PRIMARY KEY(X_Min, Y_Min, X_Max, Y_Max, Image_Name))')
            print('IMAGE table created')


    def insertImage(self, image):
        """Insert image metadata into image table

        Arguments:
            image {tuple} -- tuple containing the 4 variables of image metadata
        """
        sql = 'INSERT INTO IMAGE(Image_Name, Image_Date, Image_Time, Init_Vector) \
                                VALUES (%s, %s, %s, %s), \
                                ON DUPLICATE KEY UPDATE \
                                Image_Name=VALUES(image_name), Image_Date=VALUES(image_date),\
                                Image_Time=VALUES(image_time), Init_Vector=VALUES(init_vector)'
        
        self.mycursor.execute(sql, image)

    def insertBBox(self, bbox):
        """Insert bounding box information into bounding box table

        Arguments:
            bbox {tuple} -- tuple containing 7 variables of bounding box information
        """
        sql = 'INSERT INTO BBOX(X_Min, Y_Min, X_Max, Y_Max, Confidence, Goggles, Image_Name) \
                                VALUES (%s, %s, %s, %s, %s, %s, %s), \
                                ON DUPLICATE KEY UPDATE \
                                X_Min=VALUES(x_min), Y_Min=VALUES(y_min), \
                                X_Max=VALUES(x_max), Y_Max=VALUES(y_max), \
                                Confidence=VALUES(confidence), Goggles=VALUES(goggles), \
                                Image_Name=VALUES(image_name)'
        
        self.mycursor.execute(sql, bbox)
        
    def clearTable(self, table_name):
        """Clear input table in the MySQL database

        Arguments:
            table_name {string} -- string containing table name that needs to be cleared
        """
        try:
            sql = 'DELETE FROOM' + table_name
            self.mycursor.execute(sql)
            self.mydb.commit()
        except Exception as e:
            print(e)
            sys.exit()

