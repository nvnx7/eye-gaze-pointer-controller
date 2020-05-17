'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
'''
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':1, 'slow':10, 'medium':5}
        pyautogui.FAILSAFE=False

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def get_screen_size(self):
        return pyautogui.size()

    def move_to_center(self):
        size=self.get_screen_size()
        pyautogui.moveTo(int(size[0]/2), int(size[1]/2))

    def move(self, x, y):
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)
