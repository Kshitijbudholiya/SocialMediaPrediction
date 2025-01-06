from scripts.db import mycursor
import bcrypt

def login(username, password):
    query = f"SELECT * FROM `users` WHERE `Email` = '{username}'"
    mycursor.execute(query)
    res = mycursor.fetchall()
    numberOfUsers = len(res)
    name = ""
    if (numberOfUsers != 1):
        return "Incorrect username or password", False, name
    else:
        name = res[0][1]
        dbPass = res[0][3]
        active = res[0][5]
        checkPass = bcrypt.checkpw(password.encode(), dbPass.encode())
        if active == "Active":
            if (checkPass):
                return "Login successful!", True, name
            else:
                return "Incorrect username or password", False, name
        else:
            return "Activate your account.", False, name
