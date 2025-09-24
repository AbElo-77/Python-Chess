from flask import Flask, jsonify, request
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

@app.route("/simulate_move", methods=["POST"])
def simulate_move(): 

    data = request.get_json(); 
    board_fen = data.get("current_fen"); 
    move_made = data.get("current_move"); 

    

@app.route("/move", methods=["POST"])
def make_move():
    
    data = request.get_json(); 
    move_uci = data.get("move"); 

    try:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move); 
            return jsonify({"success": True, "board_fen": board.fen()})
        else:
            return jsonify({"success": False, "error": "Illegal Move"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
