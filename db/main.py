#from imagedb import ImageDB
from file_transfer import FTPConn

def main(): 

    #dbcon = ImageDB()
    #dbcon.init_tables()
    #dbcon.insert_image('test_image', '6-3-20', '8:46AM','asdfg')

    ftpconn = FTPConn()
    ftpconn.transfer('.', './Documents')
    ftpconn.disconnect()

if __name__ == '__main__':
    main()
