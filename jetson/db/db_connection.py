import mysql.connector
from mysql.connector import errorcode

import sys
import config

from contextlib import contextmanager, closing


#class SQLConn:

    # def __init__(self):
    #     """setup connection to MySQL database
    #     """

    #     #Specify database
    #     mydatabase = config.KEYSPACE

    #     try:
    #         self.mydb = mysql.connector.connect(
    #             host=config.SQL_HOST,
    #             user=config.USER_NAME,
    #             password=config.PASSWORD,
    #             database= mydatabase
    #         )
    #         print('Connected to mysql databse' + mydatabase)

    #     except mysql.connector.Error as err:
    #         print(str(err))
    #         sys.exit()
    #     except _mysql_connector.MySQLInterfaceError as e:
    #         print(str(e))
    #         sys.exit()
    #     except Exception as e:
    #         print(str(e))
    #         sys.exit()

    #     self.mycursor = self.mydb.cursor(buffered=True)

    # def dropImageTable(self):
    #     """Delete image table
    #     """
    #     self.mycursor.excute('DROP TABLE IF EXISTS IMAGE')
    #     print('IMAGE table dropped')

    # def dropBBoxTable(self):
    #     """Delete bounding box table
    #     """
    #     self.mycursor.execute('DROP TABLE IF EXISTS BBOX')
    #     print('BBOX table dropped')

    # def CreateImageTable(self):
    #     """Create image table
    #     """
    #     try:
    #         self.mycursor.execute('SELECT 1 FROM IMAGE LIMIT 1')
    #         print('IMAGE table exists')

    #     except:
    #         self.mycursor.execute('CREATE TABLE IMAGE(Image_Name VARCHAR(25) NOT NULL,\
    #                                 Image_Date DATE NOT NULL,\
    #                                 Image_Time TIME NOT NULL,\
    #                                 Init_Vector VARCHAR(25) NOT NULL,\
    #                                     PRIMARY KEY(Image_Name))')
    #         print('IMAGE table created')

    # def CreateBBoxTable(self):
    #     """Create bounding box table
    #     """
    #     try:
    #         self.mycursor.execute('SELECT 1 FROM BBOX LIMIT 1')
    #         print('BBOX table exists')

    #     except:
    #         self.mycursor.execute('CREATE TABLE BBOX(X_Min VARCHAR(10) NOT NULL,\
    #                                 Y_Min VARCHAR(10) NOT NULL,\
    #                                 X_Max VARCHAR(10) NOT NULL,\
    #                                 Y_Max VARCHAR(10) NOT NULL,\
    #                                 Confidence VARCHAR(10) NOT NULL,\
    #                                 Goggles VARCHAR(10) NOT NULL,\
    #                                 Image_Name VARCHAR(25) NOT NULL,\
    #                                     PRIMARY KEY(X_Min, Y_Min, X_Max, Y_Max, Image_Name))')
    #         print('IMAGE table created')

    # def insertImage(self, image):
    #     """Insert image metadata into image table

    #     Arguments:
    #         image {tuple} -- tuple containing the 4 variables of image metadata
    #     """
    #     sql = 'INSERT INTO IMAGE(Image_Name, Image_Date, Image_Time, Init_Vector) \
    #                             VALUES (%s, %s, %s, %s), \
    #                             ON DUPLICATE KEY UPDATE \
    #                             Image_Name=VALUES(image_name), Image_Date=VALUES(image_date),\
    #                             Image_Time=VALUES(image_time), Init_Vector=VALUES(init_vector)'

    #     self.mycursor.execute(sql, image)

    # def insertBBox(self, bbox):
    #     """Insert bounding box information into bounding box table

    #     Arguments:
    #         bbox {tuple} -- tuple containing 7 variables of bounding box information
    #     """
    #     sql = 'INSERT INTO BBOX(X_Min, Y_Min, X_Max, Y_Max, Confidence, Goggles, Image_Name) \
    #                             VALUES (%s, %s, %s, %s, %s, %s, %s), \
    #                             ON DUPLICATE KEY UPDATE \
    #                             X_Min=VALUES(x_min), Y_Min=VALUES(y_min), \
    #                             X_Max=VALUES(x_max), Y_Max=VALUES(y_max), \
    #                             Confidence=VALUES(confidence), Goggles=VALUES(goggles), \
    #                             Image_Name=VALUES(image_name)'

    #     self.mycursor.execute(sql, bbox)

    # def clearTable(self, table_name):
    #     """Clear input table in the MySQL database

    #     Arguments:
    #         table_name {string} -- string containing table name that needs to be cleared
    #     """
    #     try:
    #         sql = 'DELETE FROM' + table_name
    #         self.mycursor.execute(sql)
    #         self.mydb.commit()
    #     except Exception as e:
    #         print(e)
    #         sys.exit()


@contextmanager
def sql_connection():
    """Sets up connection to mysql database

    Yields:
        sql connection: sql connector object to the sql database 
    """
    connection = mysql.connector.connect(
        host=config.SQL_HOST,
        user=config.USER_NAME,
        password=config.PASSWORD,
        database=config.KEYSPACE
    )

    with closing(connection) as connection:
        yield connection


@contextmanager
def sql_cursor():
    """Gets sql cursor from database connection

    Yields:
       sql cursor : cursor object to database
    """
    with sql_connection() as connection:
        yield connection.cursor()


def sql_insert(table_name, **kwargs):
    """Inserts row of information for a specified table in database

    Args:
        table_name (string): name of table in database, must be spelt exactly as specified in database
    """
    key_list = ','.join([key for key, _ in kwargs.items()])
    value_list = ','.join([f'%({key})s' for key, _ in kwargs.items()])
    query = f"INSERT INTO %(table_name)s ({key_list}) VALUES({value_list})"
    params = kwargs

    params['table_name'] = table_name
    with sql_cursor() as cursor:
        try:
            cursor.execute(query, params)
        except Exception as e:
            print(e)

def sql_clear_table(table_name):
    """Clears all rows in specified table in the database

    Args:
        table_name (string): name of table in database, must be spelt exactly as specified in database
    """
    query = 'DELETE FROM %(table_name)s'
    params = {'table_name': table_name}
    with sql_cursor() as cursor:
        cursor.execute(query, params)

def insert_image(image_name, image_date, image_time, init_vector):
    """Inserts image metadata into IMAGE table in database

    Args:
        image_name (string): name of image
        image_date (datetime obj): date the image was taken
        image_time (datatime obj): time the image was taken
        init_vector (string): decryption string for encrypted image
    """
    sql_insert('IMAGE', Image_Name=image_name, Image_Date=image_date,
               Image_Time=image_time, Init_Vector=init_vector)


def insert_bbox(x_min, y_min, x_max, y_max, confidence, goggles, image_name):
    """Inserts bounding boxes for an images into BBOX table in database

    Args:
        x_min (float): bottom left x-coordinate for bounding box
        y_min (float): bottom left y-coordinate for bounding box
        x_max (float): top right x-coordinate for bounding box
        y_max (float): top right y-coordinate for bounding box
        confidence (float): confidence score for bounding box
        goggles (bool): whether goggles were detected in the bounding box
        image_name (string): name of image that contains the bounding box
    """
    sql_insert('BBOX', X_Min=x_min, Y_Min=y_min, X_Max=x_max,
               Y_Max=y_max, Confidence=confidence, Goggles=goggles, Image_Name=image_name)


if __name__ == '__main__':
    insert_image('testimage', '100', '123', '232141fs')