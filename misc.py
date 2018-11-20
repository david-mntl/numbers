import os
from tkinter import PhotoImage


'''
*   Loads an images file from the imgs directory
*   @param pName: name of the image relative to the imgs directory
'''
def load_img(pName):
    path = os.path.join("imgs",pName)
    img = PhotoImage(file = path)
    return img

'''
*   GUI Constantes for colors
'''
lightBlue = "#B1B1CB"
hardBlue = "#546DD0"
black = "#000000"
green = "#80BB7A"
red = "#D06F6F"