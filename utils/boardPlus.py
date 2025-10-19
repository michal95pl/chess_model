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

    def get_move_index(self, move: chess.Move):
        piece = self.piece_at(move.from_square)

        # promotion
        if move.promotion:
            # 0 - left, 1 - forward, 2 - right
            move_index = move.to_square - move.from_square - 7
            if move.promotion == chess.KNIGHT:
                return 84 + move_index
            if move.promotion == chess.BISHOP:
                return 87 + move_index
            if move.promotion == chess.ROOK:
                return 90 + move_index
            if move.promotion == chess.QUEEN:
                return 93 + move_index

        # castling
        if self.is_kingside_castling(move):
            return 96
        if self.is_queenside_castling(move):
            return 97

        if piece.piece_type == chess.PAWN:
            # 1 square forward
            if abs(move.from_square - move.to_square) == 8:
                return 0
            # 2 squares forward
            if abs(move.from_square - move.to_square) == 16:
                return 1
            # capture left
            if abs(move.from_square - move.to_square) == 7:
                return 2
            # capture right
            if abs(move.from_square - move.to_square) == 9:
                return 3

        elif piece.piece_type == chess.KNIGHT:
            knight_moves = {
                6: 4,
                10: 5,
                15: 6,
                17: 7,
                -6: 8,
                -10: 9,
                -15: 10,
                -17: 11
            }
            move_diff = move.from_square - move.to_square
            if move_diff in knight_moves:
                return knight_moves[move_diff]

        elif piece.piece_type == chess.BISHOP:
            # right
            if move.from_square % 8 < move.to_square % 8:
                return 12 + (move.to_square // 8)
            # left
            elif move.from_square % 8 > move.to_square % 8:
                return 20 + (move.to_square // 8)

        elif piece.piece_type == chess.ROOK:
            # horizontal
            if move.from_square // 8 == move.to_square // 8:
                return 28 + move.to_square % 8
            # vertical
            elif move.from_square % 8 == move.to_square % 8:
                return 36 + move.to_square // 8

        elif piece.piece_type == chess.QUEEN:
            # horizontal
            if move.from_square // 8 == move.to_square // 8:
                return 44 + move.to_square % 8
            # vertical
            elif move.from_square % 8 == move.to_square % 8:
                return 52 + move.to_square // 8
            # diagonal left
            elif move.from_square % 8 > move.to_square % 8:
                return 60 + (move.to_square // 8)
            # diagonal right
            elif move.from_square % 8 < move.to_square % 8:
                return 68 + (move.to_square // 8)

        # king
        elif piece.piece_type == chess.KING:
            king_moves = {
                1: 76,
                -1: 77,
                7: 78,
                -7: 79,
                8: 80,
                -8: 81,
                9: 82,
                -9: 83
            }
            move_diff = move.from_square - move.to_square
            if move_diff in king_moves:
                return king_moves[move_diff]

        raise ValueError("Invalid move: {} to {}. Piece: {}".format(
            chess.square_name(move.from_square),
            chess.square_name(move.to_square),
            chess.PIECE_NAMES[piece.piece_type]
        ))

    # works only for white perspective!
    def get_move_from_index(self, from_square: chess.Square, index) -> chess.Move:
        if 84 <= index <= 95:
            if index < 87:
                return chess.Move(from_square, from_square + 7 + (index - 84), promotion=chess.KNIGHT)
            elif index < 90:
                return chess.Move(from_square, from_square + 7 + (index - 87), promotion=chess.BISHOP)
            elif index < 93:
                return chess.Move(from_square, from_square + 7 + (index - 90), promotion=chess.ROOK)
            else:
                return chess.Move(from_square, from_square + 7 + (index - 93), promotion=chess.QUEEN)

        # kingside castling
        if index == 96:
            if from_square == chess.E1:
                return chess.Move(from_square, chess.G1)
            elif from_square == chess.D1:
                return chess.Move(from_square, chess.B1)
        # queenside castling
        if index == 97:
            if from_square == chess.E1:
                return chess.Move(from_square, chess.C1)
            elif from_square == chess.D1:
                return chess.Move(from_square, chess.F1)

        # pawn moves
        if 0 <= index <= 3:
            pawn_diffs = [8, 16, 7, 9]
            to_square = from_square + pawn_diffs[index]
            return chess.Move(from_square, to_square)

        # knight moves
        if 4 <= index <= 11:
            knight_moves = [6, 10, 15, 17, -6, -10, -15, -17]
            to_square = from_square - knight_moves[index - 4]
            return chess.Move(from_square, to_square)

        # Bishop moves
        if 12 <= index <= 19:
            # right
            rank = index - 12
            diff_rank = rank - (from_square // 8)
            x = (from_square % 8) + abs(diff_rank)
            y = (from_square // 8) + diff_rank
            return chess.Move(from_square, (8 * y) + x)
        if 20 <= index <= 27:
            # left
            rank = index - 20
            diff_rank = rank - (from_square // 8)
            x = (from_square % 8) - abs(diff_rank)
            y = (from_square // 8) + diff_rank
            return chess.Move(from_square, (8 * y) + x)

        # Rook moves
        if 28 <= index <= 35:
            # horizontal
            file = index - 28
            diff_file = file - (from_square % 8)
            return chess.Move(from_square, from_square + diff_file)
        if 36 <= index <= 43:
            # vertical
            rank = index - 36
            diff_rank = rank - (from_square // 8)
            return chess.Move(from_square, from_square + (diff_rank * 8))

        # Queen moves
        if 44 <= index <= 51:
            # horizontal
            file = index - 44
            diff_file = file - (from_square % 8)
            return chess.Move(from_square, from_square + diff_file)
        if 52 <= index <= 59:
            # vertical
            rank = index - 52
            diff_rank = rank - (from_square // 8)
            return chess.Move(from_square, from_square + (diff_rank * 8))
        if 60 <= index <= 67:
            # diagonal left
            rank = index - 60
            diff_rank = rank - (from_square // 8)
            x = (from_square % 8) - abs(diff_rank)
            y = (from_square // 8) + diff_rank
            return chess.Move(from_square, (8 * y) + x)
        if 68 <= index <= 75:
            # diagonal right
            rank = index - 68
            diff_rank = rank - (from_square // 8)
            x = (from_square % 8) + abs(diff_rank)
            y = (from_square // 8) + diff_rank
            return chess.Move(from_square, (8 * y) + x)

        # King moves
        if 76 <= index <= 83:
            king_diffs = [1, -1, 7, -7, 8, -8, 9, -9]
            to_square = from_square - king_diffs[index - 76]
            return chess.Move(from_square, to_square)

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

    def encode(self):
        """
        Encode board (8x8) to 13x8x8 array. Ech dimension contains 1 if the piece is present on the square
        """
        board_list = np.zeros((13, 8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                piece = self.piece_at(chess.square(j, 7 - i))
                if piece is not None:
                    piece_type = piece.symbol()
                    board_list[self.piece_mapper[piece_type]][i][j] = 1
                else:
                    board_list[self.piece_mapper[' ']][i][j] = 1
        return board_list

    def get_available_moves_mask(self):
        available_moves = list(self.legal_moves)
        action_vector = np.zeros((8, 8, 98), dtype=int)
        for move in available_moves:
            action_vector[move.from_square // 8][move.from_square % 8][self.get_move_index(move)] = 1
        return action_vector.flatten()

    def encode_move(self, move: chess.Move):
        """
        Return action vector of 6272 (8x8x98) elements. 1 if the move was made
        """
        action_vector = np.zeros((8, 8, 98), dtype=int)
        action_vector[move.from_square // 8][move.from_square % 8][self.get_move_index(move)] = 1
        return action_vector.flatten()

    def decode_move(self, index) -> chess.Move:
        """
        Decode action vector of 6272 (8x8x98) elements to chess.Move.
        """
        from_square = index // 98
        move_index = index % 98
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
