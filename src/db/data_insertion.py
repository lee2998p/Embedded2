from src.db.file_transfer import ftp_transfer
from src.db.db_connection import sql_insert, IMAGE, BBOX
from decimal import Decimal
import datetime

"""Only neither and goggles classifications are currently stored in the database as a bit value. Hence these variables define
the standard being used.
"""
GOGGLES = 1
NEITHER = 0

def data_insert(image_name: str, image_date: datetime, image_time: datetime, init_vecs: list, bboxes: list, input_dir: str, labels: list):
    """Transfer image to remote storage then inserts image metadata and bounding boxes data in database

    Args:
        image_name (string): name of image, should be a unique name to avoid duplicates in database
        image_date (datetime obj): date image was taken
        image_time (datetime obj): time image was taken
        init_vecs (list): list of decryption keys(strings) for encrypted bounding boxes in image
        bboxes (list): list of bounding boxes, each bounding box containing coordinates, confidence and classification
        labels (list): defines the detected classification for each bounding box
        input_dir (string): image path in client machine
    """
    
    """The 3 classifications included in labels are represented as follows:
    glasses : 0, goggles : 1, neither : 2
    Glasses classification is currently not being stored in the database so the list is modified before database insertion
    """
    labels = [GOGGLES if label == 1 else NEITHER for label in labels]

    # Below ftp transfer has been commented out for testing purposes and will be fixed by another team member
    with ftp_transfer() as transfer:
        transfer(input_dir, './Documents', image_name)

    sql_insert(IMAGE(image_name, image_date, image_time))

    for bbox, init_vec, label in zip(bboxes, init_vecs, labels):
        sql_insert(BBOX(float(bbox[0]), float(bbox[1]), float(bbox[2]), 
                        float(bbox[3]), float(bbox[4]), label, image_name, init_vec))
