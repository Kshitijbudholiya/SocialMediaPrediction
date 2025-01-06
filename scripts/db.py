import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="socialmedia"
)

mycursor = mydb.cursor()
