#!/usr/bin/env python3
"""
my_engine.py — 一个可用的 Python UCI 国际象棋引擎
依赖: pip install python-chess
"""
import sys
import time
import chess
import chess.polyglot

# ============================================================
# 1. 常量
# ============================================================
INF  = 1_000_000
MATE = 100_000

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

PST = {
    chess.PAWN: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10,-20,-20, 10, 10,  5,
          5, -5,-10,  0,  0,-10, -5,  5,
          0,  0,  0, 20, 20,  0,  0,  0,
          5,  5, 10, 25, 25, 10,  5,  5,
         10, 10, 20, 30, 30, 20, 10, 10,
         50, 50, 50, 50, 50, 50, 50, 50,
          0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}

# ============================================================
# 2. 评估函数（相对当前走棋方）
# ============================================================
def evaluate(board):
    if board.is_checkmate():
        return -MATE + board.ply()      # 越快被将杀分越低
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        v = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            v += PST[piece.piece_type][sq]
            score += v
        else:
            v += PST[piece.piece_type][chess.square_mirror(sq)]
            score -= v

    return score if board.turn == chess.WHITE else -score

# ============================================================
# 3. 走法排序（MVV-LVA）
# ============================================================
def order_moves(board):
    def score(move):
        s = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            # 吃过路兵时 to_square 为空
            victim_val = PIECE_VALUES[victim.piece_type] if victim else PIECE_VALUES[chess.PAWN]
            attacker = board.piece_at(move.from_square)
            attacker_val = PIECE_VALUES[attacker.piece_type] if attacker else 0
            s += 10 * victim_val - attacker_val
        if move.promotion:
            s += PIECE_VALUES[move.promotion]
        return s
    return sorted(board.legal_moves, key=score, reverse=True)

# ============================================================
# 4. 全局搜索状态
# ============================================================
_deadline = None
_nodes = 0

# ============================================================
# 5. 静态搜索（避免地平线效应）
# ============================================================
def quiescence(board, alpha, beta):
    global _nodes
    _nodes += 1
    if _nodes % 4096 == 0 and time.time() > _deadline:
        raise TimeoutError

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move) and not move.promotion:
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

# ============================================================
# 6. 主搜索：Negamax + Alpha-Beta + 置换表
# ============================================================
EXACT, LOWER, UPPER = 0, 1, 2

def search(board, depth, alpha, beta, tt):
    global _nodes
    _nodes += 1
    if _nodes % 2048 == 0 and time.time() > _deadline:
        raise TimeoutError

    # 置换表探测
    key = chess.polyglot.zobrist_hash(board)
    entry = tt.get(key)
    if entry is not None:
        e_depth, e_score, e_flag, _ = entry
        if e_depth >= depth:
            if e_flag == EXACT:
                return e_score
            elif e_flag == LOWER and e_score >= beta:
                return e_score
            elif e_flag == UPPER and e_score <= alpha:
                return e_score

    if board.is_game_over():
        return evaluate(board)
    if depth <= 0:
        return quiescence(board, alpha, beta)

    alpha_orig = alpha
    best_score = -INF
    best_move = None

    for move in order_moves(board):
        board.push(move)
        score = -search(board, depth - 1, -beta, -alpha, tt)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    if best_score <= alpha_orig:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    else:
        flag = EXACT
    tt[key] = (depth, best_score, flag, best_move)
    return best_score

# ============================================================
# 7. 根搜索：迭代加深
# ============================================================
def find_best_move(board, max_depth=4, time_limit=3.0):
    global _deadline, _nodes
    _deadline = time.time() + time_limit
    _nodes = 0

    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    tt = {}
    best_move = legal[0]
    start = time.time()

    for depth in range(1, max_depth + 1):
        if time.time() > _deadline:
            break

        moves = order_moves(board)
        if best_move in moves:
            moves.remove(best_move)
            moves.insert(0, best_move)

        alpha, beta = -INF, INF
        current_best = None
        current_score = -INF
        completed = True

        try:
            for move in moves:
                board.push(move)
                score = -search(board, depth - 1, -beta, -alpha, tt)
                board.pop()
                if score > current_score:
                    current_score = score
                    current_best = move
                if score > alpha:
                    alpha = score
        except TimeoutError:
            completed = False

        if current_best is not None and (completed or depth == 1):
            best_move = current_best

        if completed:
            ms = int((time.time() - start) * 1000)
            print(f"info depth {depth} score cp {current_score} "
                  f"nodes {_nodes} time {ms} pv {best_move.uci()}")
            sys.stdout.flush()
        else:
            break

    return best_move

# ============================================================
# 8. UCI 协议层
# ============================================================
def parse_position(tokens, chess960=False):
    if len(tokens) < 2:
        return chess.Board(chess960=chess960)
    if tokens[1] == "startpos":
        board = chess.Board(chess960=chess960)
        i = 2
    elif tokens[1] == "fen":
        parts, i = [], 2
        while i < len(tokens) and tokens[i] != "moves":
            parts.append(tokens[i])
            i += 1
        fen = " ".join(parts)
        # 兼容个别 GUI 在 FEN 后附加的 "chess960" 标记
        if fen.endswith(" chess960"):
            fen = fen[:-len(" chess960")]
            chess960 = True
        board = chess.Board(fen, chess960=chess960)
    else:
        return chess.Board(chess960=chess960)

    if i < len(tokens) and tokens[i] == "moves":
        for mv in tokens[i+1:]:
            try:
                board.push_uci(mv)
            except ValueError:
                break
    return board


def parse_go(tokens):
    max_depth = 4
    time_limit = 3.0
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t == "movetime" and i + 1 < len(tokens):
            time_limit = int(tokens[i+1]) / 1000.0
            i += 2
        elif t == "depth" and i + 1 < len(tokens):
            max_depth = int(tokens[i+1])
            i += 2
        elif t in ("wtime", "btime") and i + 1 < len(tokens):
            budget = int(tokens[i+1]) / 1000.0 / 30.0   # 剩余时间/30
            time_limit = max(0.2, min(5.0, budget))
            i += 2
        elif t in ("winc", "binc") and i + 1 < len(tokens):
            time_limit += int(tokens[i+1]) / 1000.0 * 0.8
            i += 2
        else:
            i += 1
    return max_depth, time_limit

def uci_loop():
    board = chess.Board()
    chess960 = False
    default_depth = 4   # ← 想要更强就改这行

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]

        try:
            if cmd == "uci":
                print("id name MyPythonAI")
                print("id author You")
                print(f"option name Depth type spin default {default_depth} min 1 max 6")
                print("option name UCI_Chess960 type check default false")
                print("uciok")
            elif cmd == "isready":
                print("readyok")
            elif cmd == "setoption":
                try:
                    if len(tokens) > 1 and tokens[1] == "UCI_Chess960":
                        idx = tokens.index("value")
                        chess960 = (tokens[idx + 1] == "true")
                    else:
                        idx = tokens.index("value")
                        default_depth = int(tokens[idx + 1])
                except (ValueError, IndexError):
                    pass
            elif cmd == "ucinewgame":
                board = chess.Board(chess960=chess960)
            elif cmd == "position":
                board = parse_position(tokens, chess960)
            elif cmd == "go":
                d, t = parse_go(tokens)
                if "depth" not in tokens:
                    d = min(d, default_depth)
                best = find_best_move(board, max_depth=d, time_limit=t)
                print(f"bestmove {best.uci()}" if best else "bestmove 0000")
            elif cmd == "quit":
                break
            # stop / ponderhit / register 等一律忽略即可
        except Exception as e:
            print(f"info string error: {e}")
            if cmd == "go":
                legal = list(board.legal_moves)
                print(f"bestmove {legal[0].uci() if legal else '0000'}")

        sys.stdout.flush()


if __name__ == "__main__":
    uci_loop()