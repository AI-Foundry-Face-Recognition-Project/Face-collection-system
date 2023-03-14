'''# In[ ]:'''
import mysql.connector
import password as pw
maxdb = mysql.connector.connect(
  host = "127.0.0.1",
  user = "root",
  password = pw.password,
  database = "world",
  )
cursor=maxdb.cursor()

cursor.execute("SELECT * FROM city")
result = cursor.fetchall()
for row in result:
    print(row)

## https://www.maxlist.xyz/2018/09/23/python_mysql/