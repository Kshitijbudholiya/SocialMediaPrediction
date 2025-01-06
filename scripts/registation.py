from flask import request
import bcrypt
import hashlib
import secrets
from scripts.db import mycursor
from scripts.emailSending import sendMail

def emailCheck(email):
    query = f"SELECT * From `users` Where `Email` = '{email}'"
    mycursor.execute(query)
    res = mycursor.fetchall()
    numOfUser = len(res)
    if (numOfUser > 0):
        return "User Already Exists", True
    else:
        return "Registration Successfull", False

def register(name, email, password):
    # Generate salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode(), salt)

    # Generating Token
    random_bytes = secrets.token_bytes(32)
    
    # Hash the random bytes to create a unique token
    activation_token = hashlib.sha256(random_bytes).hexdigest()

    # Use a parameterized query to insert values safely
    query = "INSERT INTO `users` (`Id`, `Name`, `Email`, `Password`, `Token`, `Activation`, `Date`) VALUES (NULL, %s, %s, %s, %s, 'Inactive', current_timestamp())"
    values = (name, email, hashed_password.decode(), activation_token)

    if request.is_secure:  # Checks if the connection is using HTTPS
        protocol = "https"
    else:
        protocol = "http"
    
    # Build the base URL dynamically
    base_url = f"{protocol}://{request.host}/"

    verificationLink = f"{base_url}activate?token={activation_token}"

    mycursor.execute(query, values)

    # Commit the transaction if using InnoDB
    try:
        mycursor._connection.commit()
        try:
            body = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Registration Verification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            color: #333333;
            line-height: 1.6;
            background-color: #f4f4f4;
        }
        .container {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h2 {
            color: #4CAF50;
        }
        .button {
            background-color: #4CAF50;
            color: #ffffff;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            margin: 20px 0;
            border-radius: 5px;
            font-weight: bold;
        }
        .footer {
            font-size: 0.8em;
            color: #666666;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Welcome to Social Media Prediction!</h2>
        <p>Dear [User's Name],</p>
        <p>Thank you for registering with us! To complete your registration and activate your account, please verify your email address by clicking the button below:</p>
        <p>
            <a href="[Verification Link]" class="button">Verify Email</a>
        </p>
        <p>If the button above doesn't work, you can also copy and paste the following link into your browser:</p>
        <p><a href="[Verification Link]">[Verification Link]</a></p>
        <p>If you did not create an account, you can safely ignore this email.</p>
        <p>Thank you, <br>Social Media Prediction Team</p>
        <div class="footer">
            <p>© 2024 Social Media Prediction. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
""".replace("[User's Name]", name).replace("[Verification Link]", verificationLink)
            to = [f'{email}']
            text, mailSent = sendMail(body, to)
            if mailSent == True:
                return text+', Registration Successfull', True
            else:
                return text, False
        except Exception as e:
            return "Email not sent, Try again!", False
    except Exception as e:
        return "An Error Occurred", False
