from imagedb import ImageDB

def main(): 

    ImageDB.init_tables()
    ImageDB.insert_image('test_image', '6-3-20', '8:46AM','asdfg')

if __name__ == '__main__':
    main()