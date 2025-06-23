import chess
import pygame

from PGNDataset import PGNDataset
from boardPlus import BoardPlus


class chessGUI:
    SQUARE_SIZE = 70

    def __init__(self, board: BoardPlus):
        self.computer_algorithm_listener = None
        pygame.init()
        pygame.display.set_caption("Chess")
        pygame.display.set_icon(pygame.image.load("assets/icon.png"))
        self.font = pygame.font.Font("assets/PublicPixel.ttf", 22)
        self.window = pygame.display.set_mode((620, 620))
        self.clock = pygame.time.Clock()
        self.board = board

        self.figures = {}
        self.__init_figures()

    @staticmethod
    def __load_figure(path):
        return pygame.transform.scale(pygame.image.load(path), (50, 50))

    def __init_figures(self):
        self.figures['r'] = self.__load_figure("assets/black_rook.png")
        self.figures['n'] = self.__load_figure("assets/black_knight.png")
        self.figures['b'] = self.__load_figure("assets/black_bishop.png")
        self.figures['q'] = self.__load_figure("assets/black_queen.png")
        self.figures['k'] = self.__load_figure("assets/black_king.png")
        self.figures['p'] = self.__load_figure("assets/black_pawn.png")

        self.figures['R'] = self.__load_figure("assets/white_rook.png")
        self.figures['N'] = self.__load_figure("assets/white_knight.png")
        self.figures['B'] = self.__load_figure("assets/white_bishop.png")
        self.figures['Q'] = self.__load_figure("assets/white_queen.png")
        self.figures['K'] = self.__load_figure("assets/white_king.png")
        self.figures['P'] = self.__load_figure("assets/white_pawn.png")

    def __draw_square(self, x, y, color_even, color_odd):
        x_pos = x * self.SQUARE_SIZE
        y_pos = y * self.SQUARE_SIZE
        if (x + y) % 2 == 0:
            pygame.draw.rect(self.window, color_odd, (x_pos, y_pos, self.SQUARE_SIZE, self.SQUARE_SIZE))
        else:
            pygame.draw.rect(self.window, color_even, (x_pos, y_pos, self.SQUARE_SIZE, self.SQUARE_SIZE))

    def __draw_board(self):
        for i in range(8):
            for j in range(8):
                self.__draw_square(i, j, (80, 114, 123), (52, 73, 85))

    def __draw_piece(self, figure_symbol, x, y):
        figure = self.figures[figure_symbol]
        x_pos = (self.SQUARE_SIZE / 2) - (figure.get_width() / 2) + x * self.SQUARE_SIZE
        y_pos = (self.SQUARE_SIZE / 2) - (figure.get_height() / 2) + y * self.SQUARE_SIZE
        self.window.blit(figure, (x_pos, y_pos))

    def __draw_field_labels(self):
        for i in range(8):
            text = self.font.render(str(chr(ord('A') + i)), True, (255, 255, 255))
            x_pos = (self.SQUARE_SIZE / 2) - (text.get_width() / 2) + i * self.SQUARE_SIZE
            y_pos = 8 * self.SQUARE_SIZE + 7
            self.window.blit(text, (x_pos, y_pos))

            text = self.font.render(str(8 - i), True, (255, 255, 255))
            x_pos = 8 * self.SQUARE_SIZE + 7
            y_pos = (self.SQUARE_SIZE / 2) - (text.get_height() / 2) + i * self.SQUARE_SIZE
            self.window.blit(text, (x_pos, y_pos))

    def __show_pieces(self, board):
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if piece != ' ':
                    self.__draw_piece(piece, j, i)

    def __show_available_moves(self, board: BoardPlus, x, y):
        legal_moves = list(board.legal_moves)
        self.__draw_board()
        for move in legal_moves:
            if move.from_square == chess.square(x, y):
                self.__draw_square(chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square),
                                   (250, 218, 122), (240, 160, 75))
        self.__show_pieces(PGNDataset.convert_board_to_list(board))

    make_move = False
    x_from = 0
    y_from = 0
    x_to = 0
    y_to = 0

    def __make_move(self, x, y):
        if self.make_move:
            self.x_to = x
            self.y_to = y
            move = chess.Move(chess.square(self.x_from, self.y_from), chess.square(self.x_to, self.y_to))
            if self.y_to == 7 and self.board.piece_at(chess.square(self.x_from, self.y_from)).symbol() == 'P':
                move.promotion = chess.QUEEN
            if move in self.board.legal_moves:
                self.board.push(move)
            self.__draw_board()
            self.__show_pieces(PGNDataset.convert_board_to_list(self.board))
            self.make_move = False

        # select piece to move
        if not self.make_move:
            if self.board.piece_at(chess.square(x, y)) is not None:
                self.x_from = x
                self.y_from = y
                self.__show_available_moves(self.board, x, y)
                self.make_move = True
            else:
                self.__draw_board()
                self.__show_pieces(PGNDataset.convert_board_to_list(self.board))
                self.make_move = False

    def add_computer_algorithm_listener(self, listener: callable):
        self.computer_algorithm_listener = listener

    def run(self):
        if self.computer_algorithm_listener is None:
            raise Exception("Computer algorithm listener is not set")

        running = True
        self.window.fill((53, 55, 75))
        self.__draw_board()
        self.__draw_field_labels()
        self.__show_pieces(PGNDataset.convert_board_to_list(self.board))
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    x = x // self.SQUARE_SIZE
                    y = 7 - (y // self.SQUARE_SIZE)

                    if self.board.turn:
                        self.__make_move(x, y)

                if not self.board.turn:
                    self.board.push(self.computer_algorithm_listener())
                    self.__draw_board()
                    self.__show_pieces(PGNDataset.convert_board_to_list(self.board))

                if self.board.is_checkmate() or self.board.is_stalemate():
                    print("Game Over")
                    print("Result:", self.board.result())
                    running = False

            self.clock.tick(90)
            pygame.display.update()
        pygame.quit()
