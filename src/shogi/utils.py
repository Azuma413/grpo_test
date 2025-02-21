from typing import Tuple

def markdown_to_sfen(board_md: str) -> str:
    """Convert markdown formatted shogi board to SFEN format.
    
    Args:
        board_md: Markdown string representation of shogi board.
            Example:
            | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
            |---|---|---|---|---|---|---|---|---|
            | 香 | 桂 | 銀 | 金 | 玉 | 金 | 銀 | 桂 | 香 |
            ...
    
    Returns:
        str: SFEN format string or 'startpos' for initial position.
    """
    # Extract board part from markdown
    lines = board_md.strip().split('\n')
    board_lines = []
    for line in lines:
        if '|' in line and ('香' in line or '桂' in line or '銀' in line or 
                           '金' in line or '玉' in line or '角' in line or 
                           '飛' in line or '歩' in line or '　' in line):
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            board_lines.append(cells)

    # Extract captured pieces
    hand_pieces = ""
    for line in lines:
        if "持ち駒" in line:
            if "なし" not in line:
                pieces = line.split("：")[1].strip()
                hand_pieces = pieces.replace(" ", "")
            break

    # Convert to SFEN format
    sfen_rows = []
    for row in board_lines[1:]:  # Skip header row
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
    
    # Convert captured pieces to SFEN format
    hand_sfen = "-" if not hand_pieces else _hand_to_sfen(hand_pieces)
    
    # Return 'startpos' for initial position
    if _is_initial_position(sfen_rows, hand_sfen):
        return "startpos"
    
    # Return full SFEN format
    return f"sfen {'/'.join(sfen_rows)} b {hand_sfen} 1"

def _piece_to_sfen(piece: str) -> str:
    """Convert piece kanji to SFEN notation.
    
    Args:
        piece: Kanji representation of piece.
        
    Returns:
        str: SFEN piece notation.
    """
    sfen_map = {
        "歩": "P", "香": "L", "桂": "N", "銀": "S",
        "金": "G", "角": "B", "飛": "R", "玉": "K",
        "と": "+P", "成香": "+L", "成桂": "+N", "成銀": "+S",
        "馬": "+B", "龍": "+R"
    }
    return sfen_map.get(piece, "")

def _hand_to_sfen(hand: str) -> str:
    """Convert captured pieces to SFEN format.
    
    Args:
        hand: String of captured pieces in kanji.
        
    Returns:
        str: SFEN format for captured pieces.
    """
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

def _is_initial_position(sfen_rows: list, hand_sfen: str) -> bool:
    """Check if position is the initial game position.
    
    Args:
        sfen_rows: List of SFEN format board rows.
        hand_sfen: SFEN format captured pieces.
        
    Returns:
        bool: True if position is initial game position.
    """
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
    """Convert Japanese move notation to USI format.
    
    Args:
        move: Move in Japanese notation (e.g. ７六歩).
            Format is [file][rank][piece] in kanji/full-width numbers.
            
    Returns:
        str: Move in USI format (e.g. 7g7f).
    """
    # Number conversion maps
    number_map = {
        "１": "1", "２": "2", "３": "3", "４": "4", "５": "5",
        "６": "6", "７": "7", "８": "8", "９": "9"
    }
    
    rank_map = {
        "一": "a", "二": "b", "三": "c", "四": "d", "五": "e",
        "六": "f", "七": "g", "八": "h", "九": "i"
    }

    # Convert file number
    file = number_map.get(move[0], move[0])
    
    # Convert rank to letter
    rank = rank_map.get(move[1])
    
    # Target rank is one letter before current rank
    target_rank = chr(ord(rank) - 1)
    
    # Construct USI format move
    return f"{file}{rank}{file}{target_rank}"

def sfen_to_markdown(sfen: str, hands: str) -> str:
    """Convert SFEN format position to markdown table.
    
    Args:
        sfen: Board position in SFEN format.
        hands: Captured pieces string.
        
    Returns:
        str: Markdown table representation of board.
    """
    # Extract board part
    board_part = sfen.split()[0] if len(sfen.split()) > 1 else sfen
    
    # Create table header
    markdown = "| 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |\n"
    markdown += "|---|---|---|---|---|---|---|---|---|\n"
    
    # Piece conversion map (both uppercase for your pieces and lowercase for opponent's pieces)
    piece_map = {
        'L': '香', 'N': '桂', 'S': '銀', 'G': '金', 'K': '玉', 
        'B': '角', 'R': '飛', 'P': '歩',
        'l': '香', 'n': '桂', 's': '銀', 'g': '金', 'k': '玉', 
        'b': '角', 'r': '飛', 'p': '歩',
        '+L': '成香', '+N': '成桂', '+S': '成銀', 
        '+B': '馬', '+R': '龍', '+P': 'と',
        '+l': '成香', '+n': '成桂', '+s': '成銀', 
        '+b': '馬', '+r': '龍', '+p': 'と'
    }
    
    # Convert each row
    for row in board_part.split('/'):
        markdown_row = "|"
        i = 0
        while i < len(row):
            if row[i].isdigit():
                empty_count = int(row[i])
                markdown_row += "　|" * empty_count
                i += 1
            else:
                if i + 1 < len(row) and row[i + 1] == '+':
                    piece = row[i:i+2]
                    i += 2
                else:
                    piece = row[i]
                    i += 1
                markdown_row += f" {piece_map.get(piece, piece)} |"
        
        markdown += markdown_row + "\n"
    
    # Convert and separate captured pieces
    sente_pieces = []
    gote_pieces = []
    i = 0
    while i < len(hands):
        count = 1
        # Get count if present
        while i < len(hands) and hands[i].isdigit():
            count = int(hands[i])
            i += 1
        
        if i < len(hands):
            piece = hands[i]
            piece_kanji = piece_map.get(piece.upper(), piece)
            if piece.isupper():
                sente_pieces.extend([piece_kanji] * count)
            else:
                gote_pieces.extend([piece_kanji] * count)
            i += 1
    
    # Add captured pieces
    markdown += "\n持ち駒："
    if sente_pieces:
        markdown += "\n先手：" + " ".join(sente_pieces)
    if gote_pieces:
        markdown += "\n後手：" + " ".join(gote_pieces)
    if not sente_pieces and not gote_pieces:
        markdown += "なし"
    markdown += "\n"
    
    return markdown
