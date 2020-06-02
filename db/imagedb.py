from db_connection import SQLConn

class ImageDB:

    def __init__(self):

        self.mysql = SQLConn()
    

    def init_tables(self):
        
        try:
            self.mysql.CreateImagetable()
            self.mysql.CreateBBoxtable()
        except:
            pass

    def check_header(self):
    
    def read_data(self):

    def insert_data(self):

    
    