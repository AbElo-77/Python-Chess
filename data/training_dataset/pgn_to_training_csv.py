import pandas
import chess, chess.pgn  

# chess_csv = pandas.DataFrame(columns=['game_id','white', 'black', 'white_elo', 'black_elo', 'chess_fen', 'move_made']); 
# game_file = open('./data/PGN_files/page_10.pgn'); 

# current_game = chess.pgn.read_game(game_file); 
# game_id = 0;  

# while current_game is not None:
 
#     game_id = game_id + 1; 
#     game_board = current_game.board(); 

#     for move in current_game.mainline_moves():

#         game_board.push(move); 
#         current_fen = game_board.fen(); 
#         current_move = move; 

#         headers = current_game.headers; 
#         white = headers.get("White", ""); 
#         black = headers.get("Black", ""); 
#         white_elo = headers.get("WhiteElo", ""); 
#         black_elo = headers.get("BlackElo", ""); 

#         chess_csv.loc[len(chess_csv)] = [game_id, white, black, white_elo, black_elo, current_fen, current_move.uci()];  

#     current_game = chess.pgn.read_game(game_file); 

# chess_csv.to_csv('./data/training_dataset/page_10.csv', index=False); 