from imagedb import ImageDB

def main(): 

    dbcon = ImageDB()
    dbcon.init_tables()
    dbcon.insert_image('test_image', '6-3-20', '8:46AM','asdfg')

if __name__ == '__main__':
    main()