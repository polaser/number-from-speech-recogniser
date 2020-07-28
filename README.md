# number-from-speech-recogniser
Number from speech recogniser (Russian)

1. Clone repository
2. [Download model](https://drive.google.com/file/d/17U3D9RG7BLMwqK_EC89CyzouL7IbsxjE/view?usp=sharing) and place it near the script 'main.py'
2. Install required libs
3. Run main.py with one unnamed argument - path_to_your_file


Model - resnet34 (I changed only last layer)

Model predicts one of 42 russian words, which used to translate numbers (from 1 to 999 999) to text.
