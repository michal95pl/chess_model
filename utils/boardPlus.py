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
        self.apply_transform(chess.flip_horizontal)
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
        flipped_from_square = chess.square(7 - chess.square_file(move.from_square),
                                           7 - chess.square_rank(move.from_square))
        flipped_to_square = chess.square(7 - chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square))
        return chess.Move(flipped_from_square, flipped_to_square, promotion=move.promotion, drop=move.drop)

    piece_mapper = {
        'p': 0,
        'n': 1,
        'b': 2,
        'r': 3,
        'q': 4,
        'k': 5,
        'P': 6,
        'N': 7,
        'B': 8,
        'R': 9,
        'Q': 10,
        'K': 11,
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

    def better_clean_castling_rights(self) -> chess.Bitboard:
        """
        Returns valid castling rights filtered from
        :data:`~chess.Board.castling_rights`.
        """
        if self._stack:
            # No new castling rights are assigned in a game, so we can assume
            # they were filtered already.
            return self.castling_rights

        castling = self.castling_rights & self.rooks
        white_castling = castling & chess.BB_RANK_1 & self.occupied_co[chess.WHITE]
        black_castling = castling & chess.BB_RANK_8 & self.occupied_co[chess.BLACK]

        if not self.chess960:
            # The rooks must be on a1, h1, a8 or h8.
            white_castling &= (chess.BB_A1 | chess.BB_H1)
            black_castling &= (chess.BB_A8 | chess.BB_H8)

            # The kings must be on e1 or e8.
            # modified due to vertical flip
            if not self.occupied_co[chess.WHITE] & self.kings & ~self.promoted & (chess.BB_D1 if self.changed_perspective else chess.BB_E1):
                white_castling = 0
            if not self.occupied_co[chess.BLACK] & self.kings & ~self.promoted & (chess.BB_D8 if self.changed_perspective else chess.BB_E8):
                black_castling = 0

            return white_castling | black_castling
        else:
            # The kings must be on the back rank.
            white_king_mask = self.occupied_co[chess.WHITE] & self.kings & chess.BB_RANK_1 & ~self.promoted
            black_king_mask = self.occupied_co[chess.BLACK] & self.kings & chess.BB_RANK_8 & ~self.promoted
            if not white_king_mask:
                white_castling = 0
            if not black_king_mask:
                black_castling = 0

            # There are only two ways of castling, a-side and h-side, and the
            # king must be between the rooks.
            white_a_side = white_castling & -white_castling
            white_h_side = chess.BB_SQUARES[chess.msb(white_castling)] if white_castling else 0

            if white_a_side and chess.msb(white_a_side) > chess.msb(white_king_mask):
                white_a_side = 0
            if white_h_side and chess.msb(white_h_side) < chess.msb(white_king_mask):
                white_h_side = 0

            black_a_side = black_castling & -black_castling
            black_h_side = chess.BB_SQUARES[chess.msb(black_castling)] if black_castling else chess.BB_EMPTY

            if black_a_side and chess.msb(black_a_side) > chess.msb(black_king_mask):
                black_a_side = 0
            if black_h_side and chess.msb(black_h_side) < chess.msb(black_king_mask):
                black_h_side = 0

            # Done.
            return black_a_side | black_h_side | white_a_side | white_h_side

    def result_plus(self, *, claim_draw: bool = False) -> str:
        result = super().result(claim_draw=claim_draw)
        if self.changed_perspective:
            if result == '1-0':
                return '0-1'
            elif result == '0-1':
                return '1-0'
            return '1/2-1/2'
        return result

    def better_to_chess960(self, move: chess.Move) -> chess.Move:
        if self.changed_perspective:
            if move.from_square == chess.D1 and self.kings & chess.BB_D1:
                if move.to_square == chess.B1 and not self.rooks & chess.BB_B1:
                    return chess.Move(chess.D1, chess.A1)
                elif move.to_square == chess.F1 and not self.rooks & chess.BB_F1:
                    return chess.Move(chess.D1, chess.H1)

            elif move.from_square == chess.D8 and self.kings & chess.BB_D8:
                if move.to_square == chess.B8 and not self.rooks & chess.BB_B8:
                    return chess.Move(chess.D8, chess.B8)
                elif move.to_square == chess.F8 and not self.rooks & chess.BB_F8:
                    return chess.Move(chess.D8, chess.H8)
        else:
            if move.from_square == chess.E1 and self.kings & chess.BB_E1:
                if move.to_square == chess.G1 and not self.rooks & chess.BB_G1:
                    return chess.Move(chess.E1, chess.H1)
                elif move.to_square == chess.C1 and not self.rooks & chess.BB_C1:
                    return chess.Move(chess.E1, chess.A1)

            elif move.from_square == chess.E8 and self.kings & chess.BB_E8:
                if move.to_square == chess.G8 and not self.rooks & chess.BB_G8:
                    return chess.Move(chess.E8, chess.H8)
                elif move.to_square == chess.C8 and not self.rooks & chess.BB_C8:
                    return chess.Move(chess.E8, chess.A8)

        return move

    def better_push(self, move: chess.Move) -> None:

        move = self.better_to_chess960(move)
        board_state = chess._BoardState(self)
        self.castling_rights = self.better_clean_castling_rights()  # Before pushing stack
        self.move_stack.append(
            self._from_chess960(self.chess960, move.from_square, move.to_square, move.promotion, move.drop))
        self._stack.append(board_state)

        # Reset en passant square.
        ep_square = self.ep_square
        self.ep_square = None

        # Increment move counters.
        self.halfmove_clock += 1
        if self.turn == chess.BLACK:
            self.fullmove_number += 1

        # On a null move, simply swap turns and reset the en passant square.
        if not move:
            self.turn = not self.turn
            return

        # Drops.
        if move.drop:
            self._set_piece_at(move.to_square, move.drop, self.turn)
            self.turn = not self.turn
            return

        # Zero the half-move clock.
        if self.is_zeroing(move):
            self.halfmove_clock = 0

        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]

        promoted = bool(self.promoted & from_bb)
        piece_type = self._remove_piece_at(move.from_square)
        assert piece_type is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.piece_type_at(capture_square)

        # Handle special pawn moves.
        if piece_type == chess.PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and chess.square_rank(move.from_square) == 1:
                self.ep_square = move.from_square + 8
            elif diff == -16 and chess.square_rank(move.from_square) == 6:
                self.ep_square = move.from_square - 8
            elif move.to_square == ep_square and abs(diff) in [7, 9] and not captured_piece_type:
                # Remove pawns captured en passant.
                down = -8 if self.turn == chess.WHITE else 8
                capture_square = ep_square + down
                captured_piece_type = self._remove_piece_at(capture_square)

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        castling = piece_type == chess.KING and self.occupied_co[self.turn] & to_bb
        if castling:
            if self.changed_perspective:
                a_side = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            else:
                a_side = chess.square_file(move.to_square) < chess.square_file(move.from_square)

            self._remove_piece_at(move.from_square)
            self._remove_piece_at(move.to_square)

            if self.changed_perspective:
                if a_side:
                    self._set_piece_at(chess.F1 if self.turn == chess.WHITE else chess.F8, chess.KING, self.turn)
                    self._set_piece_at(chess.E1 if self.turn == chess.WHITE else chess.E8, chess.ROOK, self.turn)
                else:
                    self._set_piece_at(chess.B1 if self.turn == chess.WHITE else chess.B8, chess.KING, self.turn)
                    self._set_piece_at(chess.C1 if self.turn == chess.WHITE else chess.C8, chess.ROOK, self.turn)
            else:
                if a_side:
                    self._set_piece_at(chess.C1 if self.turn == chess.WHITE else chess.C8, chess.KING, self.turn)
                    self._set_piece_at(chess.D1 if self.turn == chess.WHITE else chess.D8, chess.ROOK, self.turn)
                else:
                    self._set_piece_at(chess.G1 if self.turn == chess.WHITE else chess.G8, chess.KING, self.turn)
                    self._set_piece_at(chess.F1 if self.turn == chess.WHITE else chess.F8, chess.ROOK, self.turn)

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.promoted & to_bb)
            self._set_piece_at(move.to_square, piece_type, self.turn, promoted)

            if captured_piece_type:
                self._push_capture(move, capture_square, captured_piece_type, was_promoted)

                # Update castling rights.

                self.castling_rights &= ~to_bb & ~from_bb

                if piece_type == chess.KING and not promoted:
                    if self.turn == chess.WHITE:
                        self.castling_rights &= ~chess.BB_RANK_8 if self.changed_perspective else ~chess.BB_RANK_1
                    else:
                        self.castling_rights &= ~chess.BB_RANK_1 if self.changed_perspective else ~chess.BB_RANK_8
                elif captured_piece_type == chess.KING and not self.promoted & to_bb:  # todo: check if this is correct
                    if self.turn == chess.WHITE and chess.square_rank(move.to_square) == 7:
                        self.castling_rights &= ~chess.BB_RANK_8
                    elif self.turn == chess.BLACK and chess.square_rank(move.to_square) == 0:
                        self.castling_rights &= ~chess.BB_RANK_1

        # Swap turn.
        self.turn = not self.turn

    def is_kingside_castling(self, move: chess.Move) -> bool:
        """
        Checks if the given pseudo-legal move is a kingside castling move.
        """
        if self.changed_perspective:
            return self.is_castling(move) and chess.square_file(move.to_square) < chess.square_file(move.from_square)
        return self.is_castling(move) and chess.square_file(move.to_square) > chess.square_file(move.from_square)

    def is_queenside_castling(self, move: chess.Move) -> bool:
        """
        Checks if the given pseudo-legal move is a queenside castling move.
        """
        if self.changed_perspective:
            return self.is_castling(move) and chess.square_file(move.to_square) > chess.square_file(move.from_square)
        return self.is_castling(move) and chess.square_file(move.to_square) < chess.square_file(move.from_square)
