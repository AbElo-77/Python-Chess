import pymysql
import chess, chess.pgn

# chess_database = pymysql.connect(

#     host= "127.0.0.1",
#     user= "root",
#     password= "",
#     database= "chesssql"
# )

# cursor = chess_database.cursor(); 
# game_file = open('./data/PGN_files/page_10.pgn'); 
# current_game = chess.pgn.read_game(game_file); 

# while current_game is not None: 
    
#     headers = current_game.headers; 

#     event = headers.get("Event", ""); 
#     site = headers.get("Site", ""); 
#     date = headers.get("Date", None); 
#     white = headers.get("White", ""); 
#     black = headers.get("Black", ""); 
#     result = headers.get("Result", ""); 
#     white_elo = headers.get("WhiteElo", ""); 
#     black_elo = headers.get("BlackElo", ""); 
#     termination = headers.get("Termination", "");    

#     exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True); 
#     moves = current_game.accept(exporter); 

#     cursor.execute(
#         """INSERT INTO games (event, site, date, white, black, result, white_elo, black_elo, termination, moves) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", 
#         (event, site, date, white, black, result, white_elo, black_elo, termination, moves)
#     )

#     chess_database.commit(); 
#     current_game = chess.pgn.read_game(game_file); 
    