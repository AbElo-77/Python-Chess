import chess.pgn
import csv
import io, os
from multiprocessing import Pool, cpu_count

def process_game(args):
    game_id, game_text = args
    rows = []
    try:
        game = chess.pgn.read_game(io.StringIO(game_text))
        if game is None:
            return rows

        board = game.board()
        headers = game.headers
        white = headers.get("White", "")
        black = headers.get("Black", "")
        white_elo = headers.get("WhiteElo", "")
        black_elo = headers.get("BlackElo", "")

        for move in game.mainline_moves():
            board.push(move)
            rows.append([
                game_id, white, black,
                white_elo, black_elo,
                board.fen(),
                move.uci()
            ])
    except Exception as e:
        print(f"Skip")
    return rows

def read_games_text(pgn_path):
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        game_id = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_id += 1
            print(game_id)
            yield (game_id, str(game))

def main():
    input_pgns = []
    for _, _, files in os.walk(os.path.abspath('./data/PGN_files')): 
        for file in files: 
            _, ex = os.path.splitext(file)
            if ex == '.pgn': 
                input_pgns.append(r"C:\Users\abdal\OneDrive\Desktop\Python Chess\data\PGN_files" + "\\" + file)

    output_csv = "./data/training_dataset/all_files.csv"

    num_workers = max(cpu_count() - 2, 1)  

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "game_id", "white", "black",
            "white_elo", "black_elo",
            "chess_fen", "move_made"
        ])

        for input_pgn in input_pgns:
            with Pool(processes=num_workers) as pool:
                for game_rows in pool.imap_unordered(process_game, read_games_text(input_pgn), chunksize=50):
                    writer.writerows(game_rows)

if __name__ == "__main__":
    main()
