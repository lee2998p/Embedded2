from Crypto.Cipher import AES
from Crypto import Random
import numpy as np
from pbkdf2 import PBKDF2
import salt
import os
from typing import List, Set, Dict, Tuple, Optional

def test_init():
  id_info = __init__()
  assert test_salt = os.urandom(16)
  assert id_info.salt = test_salt
  
  assert test_key = PBKDF2("passphrase", self.salt).read(16)
  assert id_info.key = test_key
  
def test_encrypt():
  assert test_encrypt_image = 
  assert IV = 
def test_decrypt():
  assert test_decrypt_image = 
  assert decrypt(self,
                coordinates: List[Tuple[int]],
                image:'numpy.ndarray[numpy.ndarray[numpy.ndarray[numpy.uint8]]]',
                IV: bytes) = test_decrypt_image
