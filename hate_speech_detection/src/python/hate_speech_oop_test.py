from os import getcwd, listdir
from hate_speech_detection.src.python.hate_speech_oop import HatebaseTwitter


# Getting Davidson's Hatbase Twitter Data
hb_path = getcwd() + "/hate_speech_detection/data/HatebaseTwitter"

# Initializing the HatebaseTwitter Class
hb = HatebaseTwitter(hb_path)
hb.eda()
