from flask import Flask, render_template, request, redirect, url_for, session
from math import ceil
from scripts.predictionScript import predictionOutput
from scripts.registation import emailCheck, register
from scripts.activate import checkToken, activateAccount
from scripts.loginScript import login

app = Flask(__name__)
app.secret_key = b'\xd8\x95\x814Ij\x014S\xc6r\xbaC\x1e>N\xa0\x16d:\x8dp_\xf1'

@app.route('/')
def home():
    if 'login' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if (('login' in session) and ('name' in session)):
        name = session['name']
        if 'prediction' in session:
            predicted_share = session.get('predicted_share', None)
            predicted_likes = session.get('predicted_likes', None)
            predicted_sentiment = session.get('sentiment_label', None)
            
            if predicted_share is None or predicted_likes is None or predicted_sentiment is None:
                return render_template('index.html', name=name, errorMessage="Prediction data is missing.")
            
            return render_template('index.html', name=name, predicted_share=predicted_share, predicted_likes=predicted_likes, predicted_sentiment=predicted_sentiment)
        else:
            return render_template('index.html', name=name)
    else:
        return redirect(url_for('home'))

@app.route('/verify-login', methods=['POST'])
def verify_login():
    email = request.form.get("email")
    password = request.form.get("password")

    # Verify the email and hashed password
    myMessage, loggedIn, name = login(email, password)
    if loggedIn:
        session['login'] = True
        session['name'] = name
        return redirect(url_for('dashboard'))
    else:
        # If verification fails, reload the login page with an error message
        return render_template("login.html", errorMessage=myMessage)
    
@app.route('/verify-signup', methods=['POST'])
def verify_signup():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirmPassword = request.form.get('confirmPassword')

    message, emailExists = emailCheck(email=email)

    if emailExists:
        return render_template('login.html', errorMessage=message)
    else:
        if password == confirmPassword:
            message, success = register(name=name, email=email, password=password)
            if success:
                return render_template('login.html', successMessage=message)
        else:
            return render_template('login.html', errorMessage="Passwords do not match")
        
@app.route('/activate', methods=["GET"])
def activate():
    token = request.args.get("token")
    text, verifyToken = checkToken(token)
    if verifyToken:
        text, verifyActivation = activateAccount(token)
        if verifyActivation:
            return render_template("activate.html", message=text)
        else:
            return render_template("activate.html", message=text)
    else:
        return render_template("activate.html", message=text)

@app.route('/predict', methods=["POST"])
def result():
    text = request.form.get('text')
    hashtags = request.form.get('hashtags')
    platform = request.form.get('platform')
    country = request.form.get('country')
    followers = request.form.get('followers')

    try:
        followers = int(followers)
        predicted_share, predicted_likes, sentiment_label = predictionOutput(text=text, hashtags=hashtags, platform=platform, country=country, followers=followers)
        session['prediction'] = True
        session['sentiment_label'] = sentiment_label
        session['predicted_likes'] = ceil(int(predicted_likes))
        session['predicted_share'] = ceil(int(predicted_share))
        return redirect(url_for('dashboard'))
    except Exception as e:
        return render_template("index.html", errorMessage="Invalid followers value")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)