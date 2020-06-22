from ftp_transfer import ftp_transfer
from db_connection import sql_insert, IMAGE, BBOX
import datetime

def data_insert(image_name: str, image_date: datetime, image_time: datetime, init_vec: str, bboxes: list, input_dir: str):
    """Transfer image to remote storage then inserts image metadata and bounding boxes data in database
    
    Args:
        image_name (string): name of image, should be a unique name to avoid duplicates in database
        image_date (datetime obj): date image was taken
        image_time (datetime obj): time image was taken
        init_vec (string): decryption key for encrypted image
        bboxes (list): list of bounding boxes, each bounding box containing coordinates, confidence and classification
        input_dir (string): image path in client machine
    """

    with ftp_transfer() as transfer:
        transfer(input_dir, './Documents/', image_name)

    sql_insert(IMAGE(image_name, image_date, image_time, init_vec))

    for bbox in bboxes:
        sql_insert(BBOX(bbox[0], bbox[1], bbox[2],
                        bbox[3], bbox[4], bbox[5], image_name))
