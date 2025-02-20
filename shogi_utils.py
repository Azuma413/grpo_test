def markdown_to_sfen(board_md: str) -> str:
    """
    マークダウン形式の将棋盤をSFEN形式に変換
    Args:
        board_md: マークダウン形式の盤面文字列
    Returns:
        str: SFEN形式の局面文字列
    """
    # マークダウンの盤面部分を抽出
    lines = board_md.strip().split('\n')
    board_lines = []
    for line in lines:
        if '|' in line and ('香' in line or '桂' in line or '銀' in line or 
                           '金' in line or '玉' in line or '角' in line or 
                           '飛' in line or '歩' in line or '　' in line):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            board_lines.append(cells)

    # 持ち駒情報を抽出
    hand_pieces = ""
    for line in lines:
        if "持ち駒" in line:
            if "なし" not in line:
                pieces = line.split("：")[1].strip()
                hand_pieces = pieces.replace(" ", "")
            break

    # SFEN形式の盤面を作成
    sfen_rows = []
    for row in board_lines[1:]:  # ヘッダー行をスキップ
        empty_count = 0
        sfen_row = ""
        
        for cell in row:
            if cell == "　":
                empty_count += 1
            else:
                if empty_count > 0:
                    sfen_row += str(empty_count)
                    empty_count = 0
                sfen_row += _piece_to_sfen(cell)
        
        if empty_count > 0:
            sfen_row += str(empty_count)
        
        sfen_rows.append(sfen_row)
    
    # 持ち駒をSFEN形式に変換
    hand_sfen = "-" if not hand_pieces else _hand_to_sfen(hand_pieces)
    
    # 初期局面の場合は "startpos" を返す
    if _is_initial_position(sfen_rows, hand_sfen):
        return "startpos"
    
    # 通常のSFEN形式を返す
    return f"sfen {'/'.join(sfen_rows)} b {hand_sfen} 1"

def _piece_to_sfen(piece: str) -> str:
    """駒の文字をSFEN表記に変換"""
    sfen_map = {
        "歩": "P", "香": "L", "桂": "N", "銀": "S",
        "金": "G", "角": "B", "飛": "R", "玉": "K",
        "と": "+P", "成香": "+L", "成桂": "+N", "成銀": "+S",
        "馬": "+B", "龍": "+R"
    }
    return sfen_map.get(piece, "")

def _hand_to_sfen(hand: str) -> str:
    """持ち駒をSFEN形式に変換"""
    result = []
    count = 1
    current_piece = None
    
    for piece in hand:
        if current_piece and piece == current_piece:
            count += 1
        else:
            if current_piece:
                result.append(str(count) if count > 1 else "")
                result.append(_piece_to_sfen(current_piece).lower())
            current_piece = piece
            count = 1
    
    if current_piece:
        result.append(str(count) if count > 1 else "")
        result.append(_piece_to_sfen(current_piece).lower())
    
    return "".join(result)

def sfen_to_markdown(sfen: str) -> str:
    """
    SFEN形式の局面をマークダウン形式に変換
    Args:
        sfen: SFEN形式の局面文字列
    Returns:
        str: マークダウン形式の盤面文字列
    """
    if sfen == "startpos":
        return create_initial_board()

    # SFENの解析
    parts = sfen.split()
    if parts[0] == "sfen":
        board = parts[1]
        hand = parts[3]
    else:
        board = sfen.split()[0]
        hand = "-"

    # 駒の変換マップ
    piece_map = {
        'P': '歩', 'L': '香', 'N': '桂', 'S': '銀',
        'G': '金', 'B': '角', 'R': '飛', 'K': '玉',
        '+P': 'と', '+L': '成香', '+N': '成桂', '+S': '成銀',
        '+B': '馬', '+R': '龍'
    }

    # 盤面の変換
    rows = board.split('/')
    markdown = ["| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |",
                "|---|---|---|---|---|---|---|---|---|"]

    for row in rows:
        md_row = ["|"]
        i = 0
        while i < len(row):
            if row[i].isdigit():
                md_row.extend(["　 |"] * int(row[i]))
                i += 1
            else:
                if i + 1 < len(row) and row[i+1] == '+':
                    piece = piece_map.get(row[i:i+2], '　')
                    i += 2
                else:
                    piece = piece_map.get(row[i], '　')
                    i += 1
                md_row.append(f" {piece} |")
        markdown.append("".join(md_row))

    # 持ち駒の変換
    if hand == "-":
        markdown.append("\n持ち駒：なし")
    else:
        hand_pieces = []
        i = 0
        while i < len(hand):
            if hand[i].isdigit():
                count = int(hand[i])
                piece = piece_map.get(hand[i+1].upper(), '')
                hand_pieces.extend([piece] * count)
                i += 2
            else:
                piece = piece_map.get(hand[i].upper(), '')
                hand_pieces.append(piece)
                i += 1
        markdown.append(f"\n持ち駒：{'　'.join(hand_pieces)}")

    return "\n".join(markdown)

def create_initial_board() -> str:
    """初期局面のマークダウン文字列を生成"""
    return """| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |
| 　 | 飛 | 　 | 　 | 　 | 　 | 　 | 角 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 | 　 |
| 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 | 歩 |
| 　 | 角 | 　 | 　 | 　 | 　 | 　 | 飛 | 　 |
| 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |

持ち駒：なし"""

def _is_initial_position(sfen_rows: list, hand_sfen: str) -> bool:
    """初期局面かどうかを判定"""
    initial_sfen = [
        "LNSGKGSNL",
        "1R5B1",
        "PPPPPPPPP",
        "9",
        "9",
        "9",
        "ppppppppp",
        "1b5r1",
        "lnsgkgsnl"
    ]
    return sfen_rows == initial_sfen and hand_sfen == "-"

def move_to_usi(move: str) -> str:
    """
    将棋の指し手（例：７六歩）をUSI形式（例：7g7f）に変換
    Args:
        move: 日本語形式の指し手
    Returns:
        str: USI形式の指し手
    """
    # 数字の変換マップ
    number_map = {
        "１": "1", "２": "2", "３": "3", "４": "4", "５": "5",
        "６": "6", "７": "7", "８": "8", "９": "9"
    }
    
    # 段の変換マップ
    rank_map = {
        "一": "a", "二": "b", "三": "c", "四": "d", "五": "e",
        "六": "f", "七": "g", "八": "h", "九": "i"
    }

    # 最初の数字を変換
    file = number_map.get(move[0], move[0])
    
    # 段を変換
    rank = rank_map.get(move[1])
    
    # 移動先は同じ位置の一つ前の段
    target_rank = chr(ord(rank) - 1)
    
    # USI形式の文字列を作成
    return f"{file}{rank}{file}{target_rank}"
