import os
import io
import json
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

class DataAugmentation:
    def __init__(self, notation_file, training_folder, new_train_dataset):
        self.notation_file = notation_file
        self.training_folder = training_folder
        self.notation_list = []
        self.new_notation_list = []
        self.new_train_set_path = new_train_dataset
        if not os.path.isdir(self.new_train_set_path):
            os.mkdir(self.new_train_set_path)
            print("Creating new image set under {}".format(self.new_train_set_path))

    def load_notations(self):
        with io.open(self.notation_file,'r') as n:
            notation_content = n.readlines()
        
        for img_entry in notation_content:
            self.notation_list.append(json.loads(img_entry))

    def save_notations(self):
        with open(os.path.join(self.new_train_set_path, 'nlabel.idl'), 'w') as n:
            for item in self.new_notation_list:
                json_str = json.dumps(item)
                n.write(json_str + '\n')
        print("New Label Notation has been saved!")
    
    def image_split_middle(self, image, org_coordinate, image_size):
        left_image = image.crop((0,0,image_size['width']/2,image_size['height']))
        right_image = image.crop((image_size['width']/2,0, image_size['width'],image_size['height']))
        new_coordinate_left, new_coordinate_right = self.coordinate_split(org_coordinate, image_size)
        return [[left_image, new_coordinate_left], [right_image, new_coordinate_right]]
    
    def coordinate_split(self, org_cords, image_size):
        """
            split the notation coordinate from original to left half and right half
            :param image_size: original image size, Format {'width': 640, 'height':360}
            :param org_cords: Format of coordinate [[top_left_x, top_left_y, bottom_right_x, bottom_right_y, category],]
            :return: Transferred coordinates with same Format as input
            """
        new_cords = []
        new_cords_left = []
        new_cords_right = []
        middle = image_size['width']/2
        for cord in org_cords:
            if cord[0]<=middle<=cord[2]:
                new_cords_left.append([cord[0],cord[1], middle, cord[3], cord[4]])
                new_cords_right.append([0, cord[1], cord[2]-middle, cord[3], cord[4]])
            # all in left
            elif cord[2]<=middle:
                new_cords_left.append(cord)
            # all in right
            else:
                new_cords_right.append([cord[0]-middle, cord[1], cord[2]-middle, cord[3], cord[4]])

        return [new_cords_left, new_cords_right]

    def image_transform_mirror(self, image, org_coordinate, image_size):
        mirror_img = ImageOps.mirror(image)
        new_coordinate = self.coordinate_transform(org_coordinate, image_size)
        return mirror_img, new_coordinate
    
    def coordinate_transform(self, org_cords, image_size):
        """
            Transform the notation coordinate from original to mirror image
            :param image_size: original image size, Format {'width': 640, 'height':360}
            :param org_cords: Format of coordinate [[top_left_x, top_left_y, bottom_right_x, bottom_right_y, category],]
            :return: Transferred coordinate with same Format as input
            """
        new_cords = []
        for cord in org_cords:
            top_left_x = image_size['width'] - cord[2]
            top_left_y = cord[1]
            bottom_right_x = image_size['width'] - cord[0]
            bottom_right_y = cord[3]
            new_cords.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y, cord[4]])
        return new_cords
    
    def image_transform_blur(self, image):
        """
            generate two blur images for training set with Gaussian and MedianFilter
            :type image: image which loaded by PIL.Image.open
            :return: 2 blur images
            """
        blur_images = [image.filter(ImageFilter.GaussianBlur(1)),
                       image.filter(ImageFilter.MedianFilter(3  ))]
        return blur_images

    def new_training_set(self):
        # Image size for this project is 640*360
        image_size = {'width': 640, 'height': 360}
        for image_detail in self.notation_list:
            image_path = None
            notation = None
            image_name =None
            for key in image_detail.keys():
                image_name = key
                notation = image_detail[key]
            image_path = os.path.join(self.training_folder, image_name)
            print("Loading image {}".format(image_path))
            
            original_image = Image.open(image_path)
            mirror_image, new_notation = self.image_transform_mirror(original_image, notation, image_size)
            blur_images = self.image_transform_blur(original_image)
            split_images_with_notation = self.image_split_middle(original_image, notation, image_size)
            blur_mirror_images = self.image_transform_blur(mirror_image)
            blur_split_images_with_notation = []
            for image_i, notation_i in split_images_with_notation:
                results = self.image_transform_blur(image_i)
                blur_split_images_with_notation += list(map((lambda img: [img, notation_i]), results))
            split_mirror_images_with_notation = self.image_split_middle(mirror_image, new_notation, image_size)
            image_list = []
            image_list.append([original_image, notation])
            for blur_image in blur_images:
                image_list.append([blur_image, notation])
            image_list.append([mirror_image, new_notation])
            for blur_mirror_image in blur_mirror_images:
                image_list.append([blur_mirror_image, new_notation])
                
            image_list += blur_split_images_with_notation
            image_list += split_images_with_notation
            image_list += split_mirror_images_with_notation

            self.save_image(image_list, image_name.split('.')[0])
                            
        print(len(self.new_notation_list))
        self.save_notations()
                                
    def save_image(self, images, image_prefix):
        i = 1
        for entry in images:
            image_name = image_prefix + "-" + str(i) + ".jpg"
            image_path = os.path.join(self.new_train_set_path, image_name)
            entry[0].save(image_path)
            self.new_notation_list.append({image_name: entry[1]})
            i += 1

