import unittest
from PIL import Image

from objectDetection import *
from objectDetection.model import SSD300

class TestDetectingObjects(unittest.TestCase):

    def test_detecton(self):
        self.imgs = {
                './data/VOC2007/JPEGImages/000001.jpg': ['person', 'dog'],

                }

        for i in self.imgs:
            print(i)

        checkpoint_file = './output/checkpoint_ssd300-150.pth.tar'
        model = InferencingModel(checkpoint_file = checkpoint_file)


if __name__  == '__main__':
    unittest.main()



