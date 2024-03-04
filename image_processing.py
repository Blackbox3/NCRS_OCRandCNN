from flask import Flask, render_template, request
import cv2
import numpy as np
import copy
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the pre-trained model
model_path = r'D:\Nepali Handwritten Character\ncrs\static\best_val_acc.hdf5'  # Update with your model path
loaded_model = load_model(model_path)

devanagari_mapping = {
    '0915': 'क', '0916': 'ख', '0917': 'ग', '0918': 'घ', '0919': 'ङ', '091A': 'च', '091B': 'छ', '091C': 'ज', '091D': 'झ', '091E': 'ञ', '091F': 'ट',
    '0920': 'ठ', '0921': 'ड', '0922': 'ढ', '0923': 'ण', '0924': 'त', '0925': 'थ', '0926': 'द', '0927': 'ध', '0928': 'न', '092A': 'प',
    '092B': 'फ', '092C': 'ब', '092D': 'भ', '092E': 'म', '092F': 'य', '0930': 'र', '0932': 'ल', '0935': 'व', '0936': 'श', '0937': 'ष',
    '0938': 'स', '0939': 'ह', '0915्ष': 'क्ष', '0924्र': 'त्र', '091C्ञ': 'ज्ञ', '0030': '०', '0031': '१', '0032': '२', '0033': '३',
    '0034': '४', '0035': '५', '0036': '६', '0037': '७', '0038': '८', '0039': '९',
    0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ', 5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ', 10: 'ट',
    11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण', 15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न', 20: 'प',
    21: 'फ', 22: 'ब', 23: 'भ', 24: 'म', 25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श', 30: 'ष',
    31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ', 36: '०', 37: '१', 38: '२', 39: '३',
    40: '४', 41: '५', 42: '६', 43: '७', 44: '८', 45: '९'
}

def Sorting_Key(rect):
    # Sorting function for bounding rectangles
    global Lines, Size
    x, y, w, h = rect
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    for i, (upper, lower) in enumerate(Lines):
        if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
            return cx + ((i + 1) * Size)

def Split_Image(Image):
    # Function to split the image into words
    global Lines, Size
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    for i in range(morph.shape[0]):
        for j in range(morph.shape[1]):
            if not morph[i][j]:
                morph[i][j] = 1
    div = gray / morph
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    h_proj = np.sum(thresh, axis=1)
    upper = None
    lower = None
    Lines = []
    for i in range(h_proj.shape[0]):
        proj = h_proj[i]
        if proj != 0 and upper == None:
            upper = i
        elif proj == 0 and upper != None and lower == None:
            lower = i
            if lower - upper >= 30:
                Lines.append([upper, lower])
            upper = None
            lower = None
    if upper:
        Lines.append([upper, h_proj.shape[0] - 1])
    Size = thresh.shape[1]
    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for upper, lower in Lines:
            if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
                bounding_rects.append([x, y, w, h])
    i = 0
    Length = len(bounding_rects)
    while i < Length:
        x, y, w, h = bounding_rects[i]
        j = 0
        while j < Length:
            distancex = abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2]))
            distancey = abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3]))
            threshx = max(abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2])),
                          abs(bounding_rects[j][0] - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - (bounding_rects[i][0] + bounding_rects[i][2])))
            threshy = max(abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3])),
                          abs(bounding_rects[j][1] - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - (bounding_rects[i][1] + bounding_rects[i][3])))
            if i != j and any([all([not any([all([bounding_rects[j][1] > y + h, bounding_rects[j][1] + bounding_rects[j][3] > y + h]),
                                             all([bounding_rects[j][1] < y, bounding_rects[j][1] + bounding_rects[j][3] < y])]),
                                   not any([all([bounding_rects[j][0] > x + w, bounding_rects[j][0] + bounding_rects[j][2] > x + w]), all([bounding_rects[j][0] < x, bounding_rects[j][0] + bounding_rects[j][2] < x])])]),
                              all([distancex <= 10, bounding_rects[i][3] + bounding_rects[j][3] + 10 >= threshy]), all([bounding_rects[i][2] + bounding_rects[j][2] + 10 >= threshx, distancey <= 10])]):
                x = min(bounding_rects[i][0], bounding_rects[j][0])
                w = max(bounding_rects[i][0] + bounding_rects[i][2], bounding_rects[j][0] + bounding_rects[j][2]) - x
                y = min(bounding_rects[i][1], bounding_rects[j][1])
                h = max(bounding_rects[i][1] + bounding_rects[i][3], bounding_rects[j][1] + bounding_rects[j][3]) - y
                bounding_rects[i] = [x, y, w, h]
                del bounding_rects[j]
                i = -1
                Length -= 1
                break
            j += 1
        i += 1
    bounding_rects.sort(key=Sorting_Key)
    Words = []
    for x, y, w, h in bounding_rects:
        crop = Image[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        for i in range(morph.shape[0]):
            for j in range(morph.shape[1]):
                if not morph[i][j]:
                    morph[i][j] = 1
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.vstack(contours)
        rect = cv2.minAreaRect(contours)
        Box = cv2.boxPoints(rect)
        index = np.argmin(np.sum(Box, axis=1))
        box = []
        box.extend(Box[index:])
        box.extend(Box[0:index])
        box = np.int0(box)
        shape = (box[1][0] - box[0][0], box[3][1] - box[0][1])
        src = np.float32(box)
        dst = np.array([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.bitwise_not(cv2.warpPerspective(cv2.bitwise_not(crop), M, shape))
        Words.append(warp.copy())
    return Words


def Split(Words):
    # Function to split words into characters
    Characters = []
    for Word in Words:
        gray = cv2.cvtColor(Word, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        for i in range(morph.shape[0]):
            for j in range(morph.shape[1]):
                if not morph[i][j]:
                    morph[i][j] = 1
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)
        original_thresh = thresh.copy()
        h_proj = np.sum(thresh, axis=1)
        Max = np.max(h_proj) / 2
        upper = None
        lower = None
        for i in range(h_proj.shape[0]):
            proj = h_proj[i]
            if proj > Max and upper == None:
                upper = i
            elif proj < Max and upper != None and lower == None:
                lower = i
                break
        if lower == None or lower > int(h_proj.shape[0] / 2):
            lower = int(h_proj.shape[0] / 2)
        for row in range(max(int(h_proj.shape[0] / 4), lower)):
            thresh[row] = 0
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        bounding_rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 25 and h > 5:
                new_y = 0
                new_h = min(Word.shape[0], h + y + 3)
                bounding_rects.append([x, new_y, w, new_h])
        bounding_rects.sort(key=lambda x: x[0] + int(x[2] / 2))
        index = 0
        Length = len(bounding_rects)
        while index < (Length - 1):
            x, y, w, h = bounding_rects[index]
            x_left = max(x, bounding_rects[index + 1][0])
            y_top = max(y, bounding_rects[index + 1][1])
            x_right = min(x + w, bounding_rects[index + 1][0] + bounding_rects[index + 1][2])
            y_bottom = min(y + h, bounding_rects[index + 1][1] + bounding_rects[index + 1][3])
            intersection_area = max(0, (x_right - x_left)) * max(0, (y_bottom - y_top))
            union = float((bounding_rects[index][2] * bounding_rects[index][3]) + (bounding_rects[index + 1][2] * bounding_rects[index + 1][3]) - intersection_area)
            area_ratio = (bounding_rects[index + 1][2] * bounding_rects[index + 1][3]) / union
            ratio = (bounding_rects[index + 1][2] * bounding_rects[index + 1][3]) / (w * h)
            if bounding_rects[index + 1][3] / bounding_rects[index + 1][2] > 3 or ratio <= 0.25 or (area_ratio > 0.9 and intersection_area != 0):
                x = min(bounding_rects[index][0], bounding_rects[index + 1][0])
                w = max(bounding_rects[index][0] + bounding_rects[index][2], bounding_rects[index + 1][0] + bounding_rects[index + 1][2]) - x
                y = min(bounding_rects[index][1], bounding_rects[index + 1][1])
                h = max(bounding_rects[index][1] + bounding_rects[index][3], bounding_rects[index + 1][1] + bounding_rects[index + 1][3]) - y
                bounding_rects[index] = (x, y, w, h)
                del bounding_rects[index + 1]
                index -= 1
                Length -= 1
            index += 1
        Word_Characters = []
        for x, y, w, h in bounding_rects:
            new_x = max(0, x - 3)
            new_w = min(Word.shape[1] - new_x, w + (x - new_x) + 3)
            crop = original_thresh[y:y + h, new_x:new_x + new_w]
            h_proj = np.sum(crop, axis=1)
            padding = None
            for i in range(h_proj.shape[0]):
                proj = h_proj[i]
                if proj != 0:
                    padding = i
                    break
            new_y = padding
            new_h = min(Word.shape[0] - new_y, h + new_y + 3)
            size = max(new_w, new_h)
            Character = np.zeros((size, size, 3), np.uint8)
            Character.fill(255)
            Character[int((size - new_h) / 2):int((size + new_h) / 2), int((size - new_w) / 2):int((size + new_w) / 2)] = Word[new_y:new_y + new_h, new_x:new_x + new_w]
            Word_Characters.append(Character.copy())
        Characters.append(copy.deepcopy(Word_Characters))
    return Characters

def predict_character(character, loaded_model):
    # Convert to grayscale with white text and black background
    character_gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
    _, character_binary = cv2.threshold(character_gray, 128, 255, cv2.THRESH_BINARY)
    character_binary = cv2.bitwise_not(character_binary)  
    # Resize to 32x32 pixels
    resized_image = cv2.resize(character_binary, (32, 32))
    # Convert to PIL Image
    pil_image = Image.fromarray(resized_image)
    # Convert to grayscale
    grayscale_image = pil_image.convert('L')
    # Convert to numpy array and normalize
    img_array = img_to_array(grayscale_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  
    # Make prediction
    prediction = loaded_model.predict(img_array)
    # Convert prediction to class label
    predicted_class = np.argmax(prediction)
    return predicted_class

def map_to_devanagari(input_values):
    devanagari_characters = []
    for value in input_values:
        devanagari_char = devanagari_mapping.get(value, str(value))
        devanagari_characters.append(devanagari_char)
    return devanagari_characters

def process_image(image_path):
    # Process the image and predict characters
    # Implementation of process_image function
    image = cv2.imread(image_path)
    words = Split_Image(image)
    characters = Split(words)
    predicted_characters = []
    for word_characters in characters:
        for character in word_characters:
            predicted_class = predict_character(character, loaded_model)
            predicted_characters.append(predicted_class)
    devanagari_characters = map_to_devanagari(predicted_characters)
    return devanagari_characters

@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # Save the uploaded file
        image_path = 'uploaded_image.jpg'
        f.save(image_path)
        # Process the image and predict characters
        predicted_characters = process_image(image_path)
        # Decode the Devanagari character string and write to a text file
        decoded_characters = "".join(predicted_characters)
        with open("decoded_characters.txt", "w", encoding="utf-8") as file:
            file.write(decoded_characters)
        return render_template('result1.html', characters=decoded_characters)

if __name__ == '__main__':
    app.run(debug=True)
