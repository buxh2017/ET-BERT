import os
import sys


this_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(this_file_path)

sys.path.append(ROOT_DIR)

os.environ['ROOT_DIR'] = ROOT_DIR
os.environ['TMP_DIR'] = ROOT_DIR + "/tmp"
os.environ['DATASET_DIR'] = ROOT_DIR + "/tmp/datasets"
os.environ['MODEL_DIR'] = ROOT_DIR + "/tmp/models"

