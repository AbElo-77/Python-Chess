import pandas, pymysql 

# Return Games With A Certain Set of Parameters

chess_database = pymysql.connect(

     host= "127.0.0.1",
     user= "root",
     password= "",
     database= "chesssql"
)

cursor = chess_database.cursor(); 

def query_database(sql_query: str): 

    cursor.execute(sql_query); 
    return cursor.fetchall(); 


