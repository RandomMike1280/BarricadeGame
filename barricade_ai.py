"""
Barricade Game with Pygame UI and AI (Corrected Wall Logic)
-----------------------------------------------------------
Controls:
- LEFT CLICK: Move pawn or Place wall.
- RIGHT CLICK: Rotate wall orientation.
- 'W' Key: Toggle Move/Wall mode.
- 'R' Key: Rotate wall orientation.
- 'ESC': Quit.

Fixes:
- Walls can now touch (form T, L, or I shapes) but cannot cross (intersect).
"""

import pygame
import sys
import math
from collections import deque
from enum import Enum
from typing import List, Tuple, Set, Optional

# --- Constants & Configuration ---
BOARD_SIZE = 9
CELL_SIZE = 60
MARGIN = 40
INFO_PANEL_WIDTH = 250
WINDOW_WIDTH = BOARD_SIZE * CELL_SIZE + MARGIN * 2 + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + MARGIN * 2

# Colors
COLOR_BG = (30, 30, 35)
COLOR_GRID = (60, 60, 70)
COLOR_TEXT = (220, 220, 220)
COLOR_HIGHLIGHT_VALID = (100, 255, 100, 150)
COLOR_HIGHLIGHT_INVALID = (255, 80, 80, 150)
COLOR_WALL_PREVIEW = (200, 200, 50, 180)
COLOR_WALL_INVALID = (255, 50, 50, 180)

COLOR_RED = (220, 60, 60)
COLOR_BLUE = (60, 100, 220)
COLOR_WALL = (200, 200, 50) # Neutral yellow for walls

class Player(Enum):
    RED = 0
    BLUE = 1

    def opposite(self):
        return Player.BLUE if self == Player.RED else Player.RED

class WallOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

class BarricadeState:
    def __init__(self):
        # Pawns: (row, col). RED starts at (0, 4) (top), BLUE at (8, 4) (bottom)
        self.pawns = {Player.RED: (0, 4), Player.BLUE: (8, 4)}
        # Walls: Set of tuples (orientation, r, c)
        # HORIZONTAL (r, c): Blocks vertical movement between row r/r+1 at cols c and c+1.
        # VERTICAL (r, c): Blocks horizontal movement between col c/c+1 at rows r and r+1.
        self.walls: Set[Tuple[WallOrientation, int, int]] = set()
        self.current_player = Player.RED
        self.winner = None
        self.walls_left = {Player.RED: 10, Player.BLUE: 10}

    def copy(self):
        new_state = BarricadeState()
        new_state.pawns = dict(self.pawns)
        new_state.walls = set(self.walls)
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.walls_left = dict(self.walls_left)
        return new_state

    def get_valid_moves(self) -> List[Tuple]:
        if self.winner is not None:
            return []

        moves = []
        pr, pc = self.pawns[self.current_player]
        op = self.current_player.opposite()
        or_, oc = self.pawns[op]

        # 1. Pawn Moves
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = pr + dr, pc + dc

            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                continue

            if self._is_blocked(pr, pc, nr, nc):
                continue

            if (nr, nc) == (or_, oc):
                jr, jc = nr + dr, nc + dc
                can_jump_straight = (
                    0 <= jr < BOARD_SIZE
                    and 0 <= jc < BOARD_SIZE
                    and not self._is_blocked(nr, nc, jr, jc)
                )
                if can_jump_straight:
                    moves.append(('move', jr, jc))

                side_dirs = ((0, -1), (0, 1)) if dr else ((-1, 0), (1, 0))
                for sdr, sdc in side_dirs:
                    sr, sc = nr + sdr, nc + sdc
                    if 0 <= sr < BOARD_SIZE and 0 <= sc < BOARD_SIZE:
                        if not self._is_blocked(nr, nc, sr, sc):
                            moves.append(('move', sr, sc))
            else:
                moves.append(('move', nr, nc))

        # 2. Wall Moves
        if self.walls_left[self.current_player] > 0:
            for orient in [WallOrientation.HORIZONTAL, WallOrientation.VERTICAL]:
                for r in range(BOARD_SIZE - 1):
                    for c in range(BOARD_SIZE - 1):
                        if self._is_valid_wall_placement(orient, r, c):
                            moves.append(('wall', orient, r, c))

        return moves

    def _is_blocked(self, r1, c1, r2, c2) -> bool:
        """Check if a direct step between two adjacent cells is blocked by a wall."""
        if r1 == r2: # Horizontal move
            c_min = min(c1, c2)
            # Check for Vertical walls between these columns
            # A Vertical wall at (r, c) blocks (r,c)<->(r,c+1) and (r+1,c)<->(r+1,c+1)
            # We are moving at row r1. So we check V walls at (r1, c_min) or (r1-1, c_min)
            for wr in [r1, r1 - 1]:
                if 0 <= wr < BOARD_SIZE - 1:
                    if (WallOrientation.VERTICAL, wr, c_min) in self.walls:
                        return True
        elif c1 == c2: # Vertical move
            r_min = min(r1, r2)
            # Check for Horizontal walls between these rows
            # A Horizontal wall at (r, c) blocks (r,c)<->(r+1,c) and (r,c+1)<->(r+1,c+1)
            # We are moving at col c1. So we check H walls at (r_min, c1) or (r_min, c1-1)
            for wc in [c1, c1 - 1]:
                if 0 <= wc < BOARD_SIZE - 1:
                    if (WallOrientation.HORIZONTAL, r_min, wc) in self.walls:
                        return True
        return False

    def _is_valid_wall_placement(self, orient: WallOrientation, r: int, c: int) -> bool:
        # 1. Bounds check
        if r < 0 or r >= BOARD_SIZE - 1 or c < 0 or c >= BOARD_SIZE - 1:
            return False

        # 2. Overlap & Intersection check
        if orient == WallOrientation.HORIZONTAL:
            # Cannot overlap (fully or partially) with another horizontal wall
            if (WallOrientation.HORIZONTAL, r, c) in self.walls: return False
            if (WallOrientation.HORIZONTAL, r, c - 1) in self.walls: return False
            if (WallOrientation.HORIZONTAL, r, c + 1) in self.walls: return False
            # Cannot cross a vertical wall
            if (WallOrientation.VERTICAL, r, c) in self.walls: return False
        else: # Vertical
            # Cannot overlap (fully or partially) with another vertical wall
            if (WallOrientation.VERTICAL, r, c) in self.walls: return False
            if (WallOrientation.VERTICAL, r - 1, c) in self.walls: return False
            if (WallOrientation.VERTICAL, r + 1, c) in self.walls: return False
            # Cannot cross a horizontal wall
            if (WallOrientation.HORIZONTAL, r, c) in self.walls: return False

        # 4. Path blocking check (BFS) - Must ensure opponent still has a path to goal
        self.walls.add((orient, r, c))
        can_reach_red = self._has_path(Player.RED)
        can_reach_blue = self._has_path(Player.BLUE)
        self.walls.remove((orient, r, c))

        return can_reach_red and can_reach_blue

    def _has_path(self, player: Player) -> bool:
        start = self.pawns[player]
        target_row = 8 if player == Player.RED else 0

        queue = deque([start])
        visited = set([start])

        while queue:
            r, c = queue.popleft()
            if r == target_row:
                return True

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) not in visited:
                        if not self._is_blocked(r, c, nr, nc):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        return False

    def apply_move(self, move: Tuple) -> 'BarricadeState':
        new_state = self.copy()
        type_ = move[0]

        if type_ == 'move':
            _, r, c = move
            new_state.pawns[new_state.current_player] = (r, c)
            if r == 8 and new_state.current_player == Player.RED:
                new_state.winner = Player.RED
            elif r == 0 and new_state.current_player == Player.BLUE:
                new_state.winner = Player.BLUE
        elif type_ == 'wall':
            _, orient, r, c = move
            new_state.walls.add((orient, r, c))
            new_state.walls_left[new_state.current_player] -= 1

        new_state.current_player = new_state.current_player.opposite()
        return new_state

class BarricadeAI:
    def __init__(self, player: Player, max_depth: int = 3):
        self.player = player
        self.max_depth = max_depth

    def get_best_move(self, state: BarricadeState) -> Optional[Tuple]:
        moves = state.get_valid_moves()
        if not moves:
            return None

        best_score = -math.inf
        best_move = moves[0]
        alpha = -math.inf
        beta = math.inf

        # Simple move ordering: prioritize moves closer to goal or walls near opponent
        # For now, just iterate
        
        for move in moves:
            next_state = state.apply_move(move)
            score = self._minimax(next_state, self.max_depth - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move

    def _minimax(self, state: BarricadeState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        if state.winner is not None:
            if state.winner == self.player:
                return 1000 + depth
            else:
                return -1000 - depth

        if depth == 0:
            return self._evaluate(state)

        moves = state.get_valid_moves()
        if not moves:
            return -1000

        if is_maximizing:
            max_eval = -math.inf
            for move in moves:
                eval_score = self._minimax(state.apply_move(move), depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in moves:
                eval_score = self._minimax(state.apply_move(move), depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, state: BarricadeState) -> float:
        dist_me = self._shortest_path(state, self.player)
        dist_opp = self._shortest_path(state, self.player.opposite())

        if dist_me is None: dist_me = 100
        if dist_opp is None: dist_opp = 100

        score = (dist_opp - dist_me) * 10
        score += (state.walls_left[self.player] - state.walls_left[self.player.opposite()]) * 2
        return score

    def _shortest_path(self, state: BarricadeState, player: Player) -> Optional[int]:
        start = state.pawns[player]
        target_row = 8 if player == Player.RED else 0

        queue = deque([(start, 0)])
        visited = set([start])

        while queue:
            (r, c), dist = queue.popleft()
            if r == target_row:
                return dist

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) not in visited:
                        if not state._is_blocked(r, c, nr, nc):
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))
        return None

class BarricadeUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Barricade AI (T-Shapes Allowed)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.big_font = pygame.font.SysFont("Arial", 24, bold=True)

        self.state = BarricadeState()
        self.ai = BarricadeAI(Player.BLUE, max_depth=3)

        self.mode = 'move'
        self.wall_orientation = WallOrientation.HORIZONTAL

        self.hover_pos = None
        self.hover_wall_pos = None
        self.valid_moves_cache = self.state.get_valid_moves()

        self.message = "Your Turn (Red)"
        self.game_over = False
        self.ai_thinking = False
        self.last_click_time = 0

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            self.handle_events()
            if not self.game_over and not self.ai_thinking:
                self.update_hover()
            self.draw()
            pygame.display.flip()

            if self.ai_thinking and not self.game_over:
                best_move = self.ai.get_best_move(self.state)
                if best_move:
                    self.state = self.state.apply_move(best_move)
                    self.check_game_status()
                    if not self.game_over:
                        self.message = "Your Turn (Red)"
                        self.valid_moves_cache = self.state.get_valid_moves()
                self.ai_thinking = False

        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if self.game_over:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.state = BarricadeState()
                    self.game_over = False
                    self.message = "Your Turn (Red)"
                    self.valid_moves_cache = self.state.get_valid_moves()
                continue

            if self.ai_thinking:
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_w:
                    self.mode = 'wall' if self.mode == 'move' else 'move'
                    self.valid_moves_cache = self.state.get_valid_moves()
                if event.key == pygame.K_r:
                    self.wall_orientation = WallOrientation.VERTICAL if self.wall_orientation == WallOrientation.HORIZONTAL else WallOrientation.HORIZONTAL

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_click()
                elif event.button == 3:
                    self.wall_orientation = WallOrientation.VERTICAL if self.wall_orientation == WallOrientation.HORIZONTAL else WallOrientation.HORIZONTAL

    def get_grid_pos_from_mouse(self):
        mx, my = pygame.mouse.get_pos()
        gx = mx - MARGIN
        gy = my - MARGIN
        if 0 <= gx < BOARD_SIZE * CELL_SIZE and 0 <= gy < BOARD_SIZE * CELL_SIZE:
            col = int(gx // CELL_SIZE)
            row = int(gy // CELL_SIZE)
            return row, col
        return None

    def update_hover(self):
        mpos = pygame.mouse.get_pos()
        mx, my = mpos
        gx = mx - MARGIN
        gy = my - MARGIN

        self.hover_pos = None
        self.hover_wall_pos = None

        if 0 <= gx < BOARD_SIZE * CELL_SIZE and 0 <= gy < BOARD_SIZE * CELL_SIZE:
            col = gx / CELL_SIZE
            row = gy / CELL_SIZE

            c_col = int(col)
            c_row = int(row)
            self.hover_pos = (c_row, c_col)

            if self.mode == 'wall':
                # Calculate nearest wall anchor
                # H Wall (r, c) center approx at (r+0.5, c+0.5) in cell-float?
                # Using the rounding logic derived earlier
                h_r = int(round(row - 0.5))
                h_c = int(round(col - 0.5))
                
                v_r = int(round(row - 0.5))
                v_c = int(round(col - 0.5))

                if self.wall_orientation == WallOrientation.HORIZONTAL:
                    if 0 <= h_r < BOARD_SIZE - 1 and 0 <= h_c < BOARD_SIZE - 1:
                        self.hover_wall_pos = (WallOrientation.HORIZONTAL, h_r, h_c)
                else:
                    if 0 <= v_r < BOARD_SIZE - 1 and 0 <= v_c < BOARD_SIZE - 1:
                        self.hover_wall_pos = (WallOrientation.VERTICAL, v_r, v_c)

    def handle_click(self):
        import time
        current_time = time.time()
        if current_time - self.last_click_time < 0.1:
            return
        self.last_click_time = current_time

        if self.mode == 'move' and self.hover_pos:
            r, c = self.hover_pos
            target_move = ('move', r, c)
            if target_move in self.valid_moves_cache:
                self.state = self.state.apply_move(target_move)
                self.check_game_status()
                if not self.game_over:
                    self.ai_thinking = True
                    self.message = "AI Thinking..."
                self.valid_moves_cache = self.state.get_valid_moves()

        elif self.mode == 'wall' and self.hover_wall_pos:
            orient, r, c = self.hover_wall_pos
            target_move = ('wall', orient, r, c)
            if target_move in self.valid_moves_cache:
                self.state = self.state.apply_move(target_move)
                self.check_game_status()
                if not self.game_over:
                    self.ai_thinking = True
                    self.message = "AI Thinking..."
                self.valid_moves_cache = self.state.get_valid_moves()

    def check_game_status(self):
        if self.state.winner:
            self.game_over = True
            if self.state.winner == Player.RED:
                self.message = "You Win! Press 'R' to restart."
            else:
                self.message = "AI Wins! Press 'R' to restart."
        else:
            self.valid_moves_cache = self.state.get_valid_moves()
            if not self.valid_moves_cache:
                self.game_over = True
                self.message = f"No moves for {self.state.current_player.name}. Game Over."

    def draw(self):
        self.screen.fill(COLOR_BG)

        # Info Panel
        panel_x = MARGIN * 2 + BOARD_SIZE * CELL_SIZE
        pygame.draw.rect(self.screen, (40, 40, 50), (panel_x - 10, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))

        info_y = 20
        title = self.big_font.render("BARRICADE", True, COLOR_TEXT)
        self.screen.blit(title, (panel_x, info_y))
        info_y += 40

        status_color = COLOR_RED if self.state.current_player == Player.RED else COLOR_BLUE
        if self.ai_thinking:
            status_text = "AI Thinking..."
            status_color = (200, 200, 200)
        elif self.game_over:
            status_text = "Game Over"
            status_color = (255, 215, 0)
        else:
            status_text = f"Turn: {self.state.current_player.name}"

        txt = self.font.render(status_text, True, status_color)
        self.screen.blit(txt, (panel_x, info_y))
        info_y += 30

        mode_txt = f"Mode: {self.mode.upper()}"
        color_mode = (100, 255, 100) if self.mode == 'move' else (255, 200, 100)
        txt = self.font.render(mode_txt, True, color_mode)
        self.screen.blit(txt, (panel_x, info_y))
        info_y += 25

        if self.mode == 'wall':
            orient_txt = f"Orientation: {self.wall_orientation.name}"
            txt = self.font.render(orient_txt, True, COLOR_TEXT)
            self.screen.blit(txt, (panel_x, info_y))
            info_y += 25

        controls = [
            "Controls:",
            "- L-Click: Action",
            "- R-Click: Rotate Wall",
            "- 'W': Toggle Mode",
            "- 'R': Rotate Wall",
            "- ESC: Quit"
        ]
        for line in controls:
            txt = self.font.render(line, True, (180, 180, 180))
            self.screen.blit(txt, (panel_x, info_y))
            info_y += 20

        msg_surf = self.font.render(self.message, True, (255, 255, 255))
        self.screen.blit(msg_surf, (panel_x, WINDOW_HEIGHT - 50))

        # Board Grid
        board_rect = pygame.Rect(MARGIN, MARGIN, BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE)
        pygame.draw.rect(self.screen, COLOR_GRID, board_rect, 2)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = pygame.Rect(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if (r + c) % 2 == 0:
                    pygame.draw.rect(self.screen, (35, 35, 40), rect)
                else:
                    pygame.draw.rect(self.screen, (30, 30, 35), rect)

        # Valid Move Highlights
        if self.mode == 'move' and self.hover_pos and not self.ai_thinking:
            r, c = self.hover_pos
            rect = pygame.Rect(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if ('move', r, c) in self.valid_moves_cache:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, COLOR_HIGHLIGHT_VALID, (0, 0, CELL_SIZE, CELL_SIZE))
                self.screen.blit(s, rect.topleft)
            else:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, COLOR_HIGHLIGHT_INVALID, (0, 0, CELL_SIZE, CELL_SIZE), 3)
                self.screen.blit(s, rect.topleft)

        # Draw Walls
        for w in self.state.walls:
            orient, r, c = w
            thickness = 8
            if orient == WallOrientation.HORIZONTAL:
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + (r + 1) * CELL_SIZE - thickness // 2
                w_len = CELL_SIZE * 2
                h_len = thickness
            else:
                x = MARGIN + (c + 1) * CELL_SIZE - thickness // 2
                y = MARGIN + r * CELL_SIZE
                w_len = thickness
                h_len = CELL_SIZE * 2

            pygame.draw.rect(self.screen, COLOR_WALL, (x, y, w_len, h_len))

        # Draw Wall Preview
        if self.mode == 'wall' and self.hover_wall_pos and not self.ai_thinking:
            orient, r, c = self.hover_wall_pos
            move_tuple = ('wall', orient, r, c)

            if orient == WallOrientation.HORIZONTAL:
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + (r + 1) * CELL_SIZE - 4
                w_len = CELL_SIZE * 2
                h_len = 8
            else:
                x = MARGIN + (c + 1) * CELL_SIZE - 4
                y = MARGIN + r * CELL_SIZE
                w_len = 8
                h_len = CELL_SIZE * 2

            s = pygame.Surface((w_len, h_len), pygame.SRCALPHA)
            if move_tuple in self.valid_moves_cache:
                pygame.draw.rect(s, COLOR_WALL_PREVIEW, (0, 0, w_len, h_len))
            else:
                pygame.draw.rect(s, COLOR_WALL_INVALID, (0, 0, w_len, h_len))

            self.screen.blit(s, (x, y))

        # Draw Pawns
        for player, pos in self.state.pawns.items():
            r, c = pos
            center_x = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
            center_y = MARGIN + r * CELL_SIZE + CELL_SIZE // 2
            radius = CELL_SIZE // 3

            color = COLOR_RED if player == Player.RED else COLOR_BLUE
            pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), radius, 2)

            label = "R" if player == Player.RED else "B"
            txt = self.big_font.render(label, True, (255, 255, 255))
            text_rect = txt.get_rect(center=(center_x, center_y))
            self.screen.blit(txt, text_rect)

if __name__ == "__main__":
    game = BarricadeUI()
    game.run()
