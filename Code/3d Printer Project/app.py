from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        prediction = model.predict(np.array([[x, y, z]]))

        if prediction[0] == 1:
            result = "Yes, There might be a problem in the 3D Printer. Please check."
        else:
            result = "No, There is no problem in the 3D Printer. You can continue working."

        return render_template('predictor.html', prediction=result)
    except Exception as e:
        return render_template('predictor.html', prediction="Error: " + str(e))

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
