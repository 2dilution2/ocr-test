# STEP1 : Load Packages
import easyocr

# STEP2 : Load Inference Module 
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# STEP3 : Load Input
img_path = 'imgs/healing.jpg'

# STEP4 : Inference
result = reader.readtext(img_path)

# STEP5 : Visualize Reult
print(result)