from flask import Flask, jsonify, request
from backend.algorithmic_processing.algorithm_interface import predict_move_cnn, predict_move_rnn, predict_move_gnn
import chess

app = Flask(__name__); 

board = chess.Board(); 

@app.route("/")
def home():
    return "Python Chess Backend - Abdalla Elokely"; 

@app.route("/board")
def get_board():
    return jsonify({
        "board_fen": board.fen(),      
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None
    })
    

@app.route("/move_cnn", methods=["POST"])
def make_move():
    
    data = request.get_json(); 
    board_fen = data.get("current_fen"); 
    
    move = predict_move_cnn(board_fen); 

    try:
        move = chess.Move.from_uci(move)
        if move in board.legal_moves:
            board.push(move); 
            return jsonify({"success": True, "board_fen": board.fen()})
        else:
            return jsonify({"success": False, "error": "Illegal Move"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
