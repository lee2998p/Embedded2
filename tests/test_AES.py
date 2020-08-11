from Crypto.Cipher import AES
from Crypto import Random
import numpy as np
from pbkdf2 import PBKDF2
import salt
import os
from typing import List, Set, Dict, Tuple, Optional

def test_init():
  assert test_salt = os.urandom(16)
  assert __init__(self).salt = test_salt
  
  asssert test_key = PBKDF2("passphrase", self.salt).read(16)
  assert __init__(self).key = test_key
  
def test_encrypt():
  #Test initialization and returning variable
  #Test the loop
def test_decrypt():
  #Test initialization and returning variable
  #Test the loop
