import os
import sys
import xml.etree.ElementTree as Et
from PIL import Image


def progress_bar(name, value, endvalue, bar_length=50, width_=20): # progres bar
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width_, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    if value == endvalue:
        sys.stdout.write('\n\n')


classes = {'Wheel' : 0} # Bu dict, txte yazılacak sınıfların sıralması içindir.



path_images = r'C:\Users\baris-pc\Desktop\Yolov5\images/' # imgelerin pathi
path_xmls = r'C:\Users\baris-pc\Desktop\Yolov5\xmls/'     # xmllerin pathi
path_txt = r'C:\Users\baris-pc\Desktop\Yolov5\texts/'     #txtlerin yazılacağı path
imgeisimleri = os.listdir(path_images)


for idx, isim in enumerate(imgeisimleri):
    if isim.endswith(('.jpg', '.jpeg', '.png')):
        basename, uzanti = isim.split('.')
        progress_bar("XML to Yolo ", idx, len(imgeisimleri))
        tree = Et.parse(path_xmls + basename + '.xml')
        root = tree.getroot()
        obejcts = root.findall("object")
        etiketler = [obj[0].text for obj in obejcts]
        etiketler = set(list(classes.keys())).intersection(set(etiketler))
        img = Image.open(path_images + isim)
    
        width, height = img.size
    
        if len(etiketler) == 0:
            continue
    
        for obj in obejcts:
            x1, y1 = int(obj[4][0].text), int(obj[4][1].text)
            x2, y2 = int(obj[4][2].text), int(obj[4][3].text)
            if obj[0].text in classes:
      
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                width_norm = (x2 - x1) / width
                height_norm = (y2 - y1) / height
                with open(path_txt + basename + '.txt', 'a') as f:
                    f.write('{:.0f} {:.6f} {:.6f} {:.6f} {:.6f} \n'.format(classes[obj[0].text], x_center, y_center, width_norm, height_norm))

print()
