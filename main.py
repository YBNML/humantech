'''
1. For the 28th Samsung Human Tech
'''

from utils_Input import Image_load

class HumanTech():
    def __init__(self,a):
        '''
        class declaration
        '''
        self.input = Image_load()


    def input_data(self):
        self.left_rgb, self.right_rgb   = self.input.load_rgb()
        self.left_gt, self.right_gt     = self.input.load_gt()
        self.left_mde, self.right_mde   = self.input.load_mde()

    def dos

if __name__ == '__main__':
    try:
        ht = HumanTech("asdf")
        ht.input_data()
        # while True:

        print("")
    finally:
        print("\n\n// END //\n")
