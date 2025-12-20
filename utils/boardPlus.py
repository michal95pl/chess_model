import chess
import numpy as np


class BoardPlus(chess.Board):

    def __init__(self, fen=None, chess960=False, changed_perspective=False):
        if fen is not None:
            super().__init__(fen, chess960=chess960)
        else:
            super().__init__(chess960=chess960)
        self.row_count = 8
        self.column_count = 8
        self.action_size = 6272
        self.changed_perspective = changed_perspective

    def __copy__(self):
        new_board = BoardPlus(self.fen())
        new_board.changed_perspective = self.changed_perspective
        return new_board

    def get_move_id(self, move: chess.Move):
        piece = self.piece_at(move.from_square)

        # promotion
        if move.promotion:
            # 0 - left, 1 - forward, 2 - right
            move_index = move.to_square - move.from_square - 7
            if move.promotion == chess.KNIGHT:
                return 64 + move_index
            if move.promotion == chess.BISHOP:
                return 67 + move_index
            if move.promotion == chess.ROOK:
                return 70 + move_index
            if move.promotion == chess.QUEEN:
                return 73 + move_index

        # castling
        if self.is_kingside_castling(move):
            return 76
        if self.is_queenside_castling(move):
            return 77

        if piece.piece_type == chess.KNIGHT:
            knight_moves = {
                6: 0,
                10: 1,
                15: 2,
                17: 3,
                -6: 4,
                -10: 5,
                -15: 6,
                -17: 7
            }
            move_diff = move.from_square - move.to_square
            if move_diff in knight_moves:
                return knight_moves[move_diff]

        if piece.piece_type == chess.QUEEN or piece.piece_type == chess.ROOK or piece.piece_type == chess.BISHOP or piece.piece_type == chess.KING or piece.piece_type == chess.PAWN:
            # right
            if (move.to_square % 8) > (move.from_square % 8) and (move.to_square // 8) == (move.from_square // 8):
                move_diff = move.to_square - move.from_square # max 7
                return move_diff + 7 # [8, 14]
            # left
            if (move.to_square % 8) < (move.from_square % 8) and (move.to_square // 8) == (move.from_square // 8):
                move_diff = move.from_square - move.to_square # max 7
                return move_diff + 14 # [15, 21]
            # up
            if (move.to_square % 8) == (move.from_square % 8) and (move.to_square // 8) > (move.from_square // 8):
                move_diff = (move.to_square - move.from_square) // 8 # max 7
                return move_diff + 21 # [22, 28]
            # down
            if (move.to_square % 8) == (move.from_square % 8) and (move.to_square // 8) < (move.from_square // 8):
                move_diff = (move.from_square - move.to_square) // 8 # max 7
                return move_diff + 28 # [29, 35]
            # diagonal right up
            if (move.to_square % 8) > (move.from_square % 8) and (move.to_square // 8) > (move.from_square // 8):
                move_diff = (move.to_square % 8 - move.from_square % 8)
                return move_diff + 35 # [36, 42]
            # diagonal right down
            if (move.to_square % 8) > (move.from_square % 8) and (move.to_square // 8) < (move.from_square // 8):
                move_diff = (move.to_square % 8 - move.from_square % 8)
                return move_diff + 42 # [43, 49]
            # diagonal left up
            if (move.to_square % 8) < (move.from_square % 8) and (move.to_square // 8) > (move.from_square // 8):
                move_diff = (move.from_square % 8 - move.to_square % 8)
                return move_diff + 49  # [50, 56]
            # diagonal left down
            if (move.to_square % 8) < (move.from_square % 8) and (move.to_square // 8) < (move.from_square // 8):
                move_diff = (move.from_square % 8 - move.to_square % 8)
                return move_diff + 56  # [57, 63]


        raise ValueError("Invalid move: {} to {}. Piece: {}".format(
            chess.square_name(move.from_square),
            chess.square_name(move.to_square),
            chess.PIECE_NAMES[piece.piece_type]
        ))

    # works only for white perspective!
    def get_move_from_index(self, from_square: chess.Square, index) -> chess.Move:
        if 64 <= index <= 75:
            if index < 67:
                return chess.Move(from_square, from_square + 7 + (index - 64), promotion=chess.KNIGHT)
            elif index < 70:
                return chess.Move(from_square, from_square + 7 + (index - 67), promotion=chess.BISHOP)
            elif index < 73:
                return chess.Move(from_square, from_square + 7 + (index - 70), promotion=chess.ROOK)
            else:
                return chess.Move(from_square, from_square + 7 + (index - 73), promotion=chess.QUEEN)

        # kingside castling
        if index == 76:
            if from_square == chess.E1:
                return chess.Move(from_square, chess.G1)
            elif from_square == chess.D1:
                return chess.Move(from_square, chess.B1)
        # queenside castling
        if index == 77:
            if from_square == chess.E1:
                return chess.Move(from_square, chess.C1)
            elif from_square == chess.D1:
                return chess.Move(from_square, chess.F1)

        # knight moves
        if 0 <= index <= 7:
            knight_moves = [6, 10, 15, 17, -6, -10, -15, -17]
            to_square = from_square - knight_moves[index]
            return chess.Move(from_square, to_square)

        # right
        if 8 <= index <= 14:
            move_diff = index - 7
            return chess.Move(from_square, from_square + move_diff)
        # left
        if 15 <= index <= 21:
            move_diff = index - 14
            return chess.Move(from_square, from_square - move_diff)
        # up
        if 22 <= index <= 28:
            move_diff = index - 21
            return chess.Move(from_square, from_square + move_diff * 8)
        # down
        if 29 <= index <= 35:
            move_diff = index - 28
            return chess.Move(from_square, from_square - move_diff * 8)
        # diagonal right up
        if 36 <= index <= 42:
            move_diff = index - 35
            return chess.Move(from_square, from_square + move_diff * 9)
        # diagonal right down
        if 43 <= index <= 49:
            move_diff = index - 42
            return chess.Move(from_square, from_square - move_diff * 7)
        # diagonal left up
        if 50 <= index <= 56:
            move_diff = index - 49
            return chess.Move(from_square, from_square + move_diff * 7)
        # diagonal left down
        if 57 <= index <= 63:
            move_diff = index - 56
            return chess.Move(from_square, from_square - move_diff * 9)


        raise ValueError("Invalid move index: {}".format(index))

    def change_perspective(self):
        self.apply_mirror()
        self.changed_perspective = not self.changed_perspective

    def compare_boards(self, other_board: chess.Board) -> bool:
        for square in chess.SQUARES:
            if self.piece_at(square) != other_board.piece_at(square):
                return False
        return True

    @staticmethod
    def show_bitboard(bitboard: int):
        temp = []
        for rank in range(8):
            row = ""
            for file in range(8):
                square = rank * 8 + file
                if (bitboard >> square) & 1:
                    row += "1 "
                else:
                    row += "0 "
            temp.append(row.strip())
        temp.reverse()
        print("\n".join(temp))

    @staticmethod
    def change_move_perspective(move: chess.Move) -> chess.Move:
        flipped_from_square = chess.square(chess.square_file(move.from_square),
                                           7 - chess.square_rank(move.from_square))
        flipped_to_square = chess.square(chess.square_file(move.to_square), 7- chess.square_rank(move.to_square))
        return chess.Move(flipped_from_square, flipped_to_square, promotion=move.promotion, drop=move.drop)

    piece_mapper = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
        ' ': 12
    }

    @staticmethod
    def get_empty_board_with_piece_index():
        return np.full((8, 8), 12)

    def get_board_with_piece_index(self) -> np.ndarray:
        """
        Encode board (8x8) to 8x8 array with numbers representing pieces (piece_mapper)
        """
        board_list = np.zeros((8, 8), dtype=np.uint8)
        for i in range(8):
            for j in range(8):
                piece = self.piece_at(chess.square(j, 7 - i))
                if piece is not None:
                    piece_type = piece.symbol()
                    board_list[i][j] = self.piece_mapper[piece_type]
                else:
                    board_list[i][j] = 12
        return board_list

    def encode(self) -> np.ndarray:
        """
        One hot encode board to (8x8x13)=832 array
        """
        board_array = self.get_board_with_piece_index()
        one_hot = np.eye(13, dtype=np.uint8)[board_array]
        return one_hot

    def get_available_moves_mask(self):
        available_moves = list(self.legal_moves)
        action_vector = np.zeros((8, 8, 78), dtype=int)
        for move in available_moves:
            action_vector[move.from_square // 8][move.from_square % 8][self.get_move_id(move)] = 1
        return action_vector.flatten()

    def get_move_index(self, move: chess.Move):
        """
        Encode move to move_id (0-4991).
        Works only for white perspective!
        """
        return self.get_move_id(move) + (move.from_square * 78)

    def encode_move(self, move_index) -> np.ndarray:
        """
        Return action vector of 4992 (8x8x78) elements. 1 if the move was made
        """
        action_vector = np.zeros(4992, dtype=np.uint8)
        action_vector[move_index] = 1
        return action_vector

    def decode_move(self, index) -> chess.Move:
        """
        Decode move_id (0-4991) to chess.Move
        """
        from_square = index // 78
        move_index = index % 78
        from_square = chess.square(from_square % 8, from_square // 8)
        return self.get_move_from_index(from_square, move_index)

    def result_plus(self, *, claim_draw: bool = False) -> str:
        result = super().result(claim_draw=claim_draw)
        if self.changed_perspective:
            if result == '1-0':
                return '0-1'
            elif result == '0-1':
                return '1-0'
            return '1/2-1/2'
        return result