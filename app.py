from tokenize import Double
from flask import Flask,request, jsonify, url_for, redirect, render_template
import pickle
import numpy as np
import textwrap
import google.generativeai as genai
from IPython.display import Markdown
import requests

# from flask import Flask, render_template, request, jsonify
import base64
import cv2
# import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("webcam.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/donate')
def donate():
    return render_template("checkout.html")

@app.route('/webinar')
def webinar():
    return render_template("event.html")

@app.route('/registration')
def Volunteer():
    return render_template("registration.html")

@app.route('/volunteer')
def Volant():
    return render_template("volunteer.html")

@app.route('/tracking')
def track():
    return render_template("tracking.html")

@app.route('/food')
def food():
    return render_template("food.html")

@app.route('/clothes')
def clothes():
    return render_template("clothes.html")

@app.route('/firstaid')
def firstaid():
    return render_template("firstaid.html")

@app.route('/thankyou')
def thank():
    return render_template("thankyou.html")

@app.route('/tsunami')
def tsunami():
    return render_template("tsunami.html")

@app.route('/earthquake')
def quake():
    return render_template("earthquake.html")

@app.route('/flood')
def flood():
    return render_template("flood.html")

@app.route('/create')
def create():
    return render_template("create.html")

@app.route('/webin')
def webianr():
    return render_template("webinar.html")


@app.route('/earthquake-prediction')
def Prediction():
    return render_template("ml.html")

@app.route('/shop')
def shop():
    return render_template("ecommerce.html")

@app.route('/learn')
def learn():
    return render_template("learn.html")

@app.route('/past')
def past():
    return render_template('past.html')

@app.route('/learn-earthquake')
def leanearth():
    return render_template("learn-earthquake.html")

@app.route('/learn-flood')
def leanflood():
    return render_template("learn-flood.html")

@app.route('/learn-landslide')
def leanslide():
    return render_template("learn-landslide.html")

GOOGLE_API_KEY = 'AIzaSyAKLppD27-OIWPINzG8nRcErZd4_bQmK-w'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-pro")

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

@app.route('/ask')
def index():
    return render_template('ask.html')

@app.route('/generate', methods=['POST'])
def generate():
    query = request.form['query']
    response = model.generate_content(query)
    markdown_text = to_markdown(response.text)
    html_text = markdown_text.data.replace('<blockquote>', '').replace('</blockquote>', '')
    return html_text

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/save_image', methods=['POST'])
def save_image():
    image_base64 = request.form['image_base64']
    image_data = base64.b64decode(image_base64.split(',')[1])
    
    # Decode the image data and convert it to OpenCV format
    nparr = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # If faces are detected, save the image
        with open('captured_image.png', 'wb') as f:
            f.write(image_data)
        return "Image saved successfully!"
    else:
        # If no faces are detected, return an error message
        return "No face detected in the image!"


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    print(prediction)
    output='{0:.{1}f}'.format(prediction[0], 2)


    if float(output)<3: 
        return render_template("ml.html", pred="The maximum magnitude of an earthquake possible at this location is: "+output+"\nThe earthquake is tolerable. It will only have small virations. \nRISK is LOW!")
    elif float(output)>6:
        return render_template("ml.html", pred="The maximum magnitude of an earthquake possible at this location is: "+output+"\nThe earthquake is very sever. High risk of loss of life and property. \nRISK is VERY HIGH!")
    else:
        return render_template("ml.html", pred="The maximum magnitude of an earthquake possible at this location is: "+output+"\nThe earthquake is sever. It can do considerable damage so plan have precaustionary steps ready. RISK is MEDIUM!")
    

def get_last_earthquake(location):
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&orderby=time&minmagnitude=1&limit=1&starttime=1970-01-01&endtime=now&latitude={location['latitude']}&longitude={location['longitude']}&maxradiuskm={location['radius']}"
    response = requests.get(url)
    data = response.json()
    if 'features' in data and len(data['features']) > 0:
        last_earthquake = data['features'][0]
        return last_earthquake
    else:
        return None



@app.route('/earthquake', methods=['POST'])
def earthquake():
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    radius = float(request.form['radius'])

    location = {
        'latitude': latitude,
        'longitude': longitude,
        'radius': radius
    }

    last_earthquake = get_last_earthquake(location)

    if last_earthquake:
        magnitude = last_earthquake['properties']['mag']
        location = last_earthquake['properties']['place']
        time = last_earthquake['properties']['time']
        return f"Last earthquake:<br>Magnitude: {magnitude}<br>Location: {location}<br>Time: {time}"
    else:
        return "No earthquake data available for the specified location."



if __name__ == '__main__':
    app.run(debug = True)