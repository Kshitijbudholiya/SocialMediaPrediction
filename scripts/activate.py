from scripts.db import mycursor

def checkToken(token):
    query = f"SELECT * From `users` Where `Token` = '{token}'"
    mycursor.execute(query)
    res = mycursor.fetchall()
    numberOfUser = len(res)
    if (numberOfUser != 1):
        return "Error Token Not Exists", False
    else:
        return "Token Exists", True
    
def activateAccount(token):
    query = f"UPDATE `users` SET `Activation` = 'Active' WHERE `users`.`Token` = '{token}'"
    try:
        mycursor.execute(query)
        mycursor._connection.commit()
        return "Congratulation account activated, You can login now", True
    except Exception as e:
        return "Account Not Activated, Try Again", False
