from flask import Flask, jsonify, request
from backend.algorithmic_processing.algorithm_interface import predict_move_cnn, predict_move_rnn, predict_move_gnn
import chess, flask_cors



app = Flask(__name__); 

flask_cors.CORS(app);  

@app.route("/")
def home():
    return "Python Chess Backend - Abdalla Elokely"; 

# @app.route("/board")
# def get_board():
#     payload = {
#         "board_fen": board.fen(),
#         "is_game_over": board.is_game_over(),
#         "result": board.result() if board.is_game_over() else None
#     }
    
#     return jsonify(payload); 

@app.route("/make_move_user", methods=["POST"])
def make_user_move():
    data = request.get_json(); 
    board = chess.Board(data.get("current_fen")); 
    move_str = data.get("move"); 

    if not move_str:
        return jsonify({"success": False, "error": "No move provided"}); 

    try:
        move_obj = chess.Move.from_uci(move_str)
        if move_obj in board.legal_moves:
            captured_piece_symbol = None; 

            if board.is_capture(move_obj):
                captured_piece = board.piece_at(move_obj.to_square); 
                captured_piece_symbol = captured_piece.symbol() if captured_piece else None; 

            board.push(move_obj);  
            payload = {"success": True, "board_fen": board.fen()}; 

            if captured_piece_symbol is not None:
                payload["capture"] = captured_piece_symbol; 

            return jsonify(payload); 
        else:
            return jsonify({
                "success": False,
                "error": f"Illegal Move - {data.get('current_fen')} - {move_str}"
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}); 


@app.route("/move_cnn", methods=["POST"])
def make_move_cnn():
    data = request.get_json(); 
    board = chess.Board(data.get("current_fen")); 
    board_fen = data.get("current_fen"); 

    move = predict_move_cnn(board_fen); 

    try:
        move_obj = chess.Move.from_uci(move); 
        if move_obj in board.legal_moves:
            captured_piece_symbol = None; 

            if board.is_capture(move_obj):
                captured_piece = board.piece_at(move_obj.to_square); 
                captured_piece_symbol = captured_piece.symbol() if captured_piece else None; 

            board.push(move_obj); 
            payload = {"success": True, "board_fen": board.fen()}; 

            if captured_piece_symbol is not None:
                payload["capture"] = captured_piece_symbol; 

            return jsonify(payload); 
        else:
            payload = {"success": False, "error": "Illegal Move"}; 
            return jsonify(payload); 
    except Exception as e:
        payload = {"success": False, "error": str(e)}; 
        return jsonify(payload); 

@app.route("/move_rnn", methods=["POST"])
def make_move_rnn():
    data = request.get_json(); 
    board = chess.Board(data.get("current_fen")); 
    board_fen = data.get("current_fen"); 

    move = predict_move_rnn(board_fen); 

    try:
        move_obj = chess.Move.from_uci(move); 
        if move_obj in board.legal_moves:
            captured_piece_symbol = None; 

            if board.is_capture(move_obj):
                captured_piece = board.piece_at(move_obj.to_square); 
                captured_piece_symbol = captured_piece.symbol() if captured_piece else None; 

            board.push(move_obj); 
            payload = {"success": True, "board_fen": board.fen()}; 

            if captured_piece_symbol is not None:
                payload["capture"] = captured_piece_symbol; 

            return jsonify(payload); 
        else:
            payload = {"success": False, "error": "Illegal Move"}; 
            return jsonify(payload); 
    except Exception as e:
        payload = {"success": False, "error": str(e)}; 
        return jsonify(payload); 

@app.route("/move_gnn", methods=["POST"])
def make_move_gnn():
    data = request.get_json(); 
    board = chess.Board(data.get("current_fen")); 
    board_fen = data.get("current_fen"); 

    move = predict_move_gnn(board_fen); 

    try:
        move_obj = chess.Move.from_uci(move); 
        if move_obj in board.legal_moves:
            captured_piece_symbol = None; 

            if board.is_capture(move_obj):
                captured_piece = board.piece_at(move_obj.to_square); 
                captured_piece_symbol = captured_piece.symbol() if captured_piece else None; 

            board.push(move_obj); 
            payload = {"success": True, "board_fen": board.fen()}; 

            if captured_piece_symbol is not None:
                payload["capture"] = captured_piece_symbol; 

            return jsonify(payload); 
        else:
            payload = {"success": False, "error": "Illegal Move"}; 
            return jsonify(payload); 
    except Exception as e:
        payload = {"success": False, "error": str(e)}; 
        return jsonify(payload); 


if __name__ == "__main__":
    app.run(debug=True)
