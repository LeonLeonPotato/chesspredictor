import time

with open("C:\\Users\\Leon\\Desktop\\projects\\ucla\\rawdata.txt", "r") as f:
    data = f.readlines()[5:]

@lambda f:f()
def process():
    processed = open("chess_processed_small.csv", "w")

    def convert_to_pgn(move_sequence):
        moves = move_sequence.split(" ")
        pgn_moves = []
        for i in range(0, len(moves), 2):
            move_number = i // 2 + 1
            white_move = moves[i].split('.')[1]
            black_move = moves[i+1].split('.')[1] if i + 1 < len(moves) and '.' in moves[i+1] else ''
            pgn_moves.append(f"{move_number}.{white_move} {black_move}")
        return " ".join(pgn_moves)

    _time = time.time()
    _processed = 0
    _total = 0
    processed.write("winner,white_elo,black_elo,diff,pgn\n")
    processed_cache = ""
    for line in data:
        metadata, pgn = line.split(" ### ")
        pgn = pgn.strip()
        metadata = metadata.strip()
        metadata = metadata.split(" ")
        winner = metadata[2]
        if "blen_false" in metadata[15] and metadata[3] != "None" and metadata[4] != "None":
            welo, belo = int(metadata[3]), int(metadata[4])
            if pgn.endswith("#"):
                if welo < 2000 or belo < 2000:
                    continue
            elif winner == "1/2-1/2":
                if welo < 2500 or belo < 2500:
                    continue
            else: continue

            pgn = convert_to_pgn(pgn)
            if winner == "1-0": 
                winner = 1
            elif winner == "0-1": 
                winner = -1
            else: 
                winner = 0
            processed_cache += f"{winner},{welo},{belo},{welo - belo},{pgn}\n"
            _processed += 1
        _total += 1

        if time.time() - _time > 0.2:
            print(f"{_processed} / {_total}")
            _time = time.time()

        if _processed > 5000:
            break

    print("done")
    processed.write(processed_cache)
    processed.close()