"""
Barricade Game with Pygame UI and AI
-------------------------------------
Controls:
- LEFT CLICK:
    - Move pawn to target square (if valid).
    - Place wall (if in Wall Mode).
- RIGHT CLICK: Toggle wall orientation (Horizontal <-> Vertical).
- 'W' Key: Toggle between Move Mode and Wall Mode.
- 'R' Key: Toggle wall orientation.
- 'F' Key: Toggle who makes the first move and restart.
- 'ESC': Quit.

Features:
- Transparent preview of wall placement on hover.
- Red highlight for invalid moves/walls.
- Green highlight for valid moves.
- AI opponent using Minimax with Alpha-Beta pruning.
"""

import pygame
import sys
import math
from collections import deque
from enum import Enum
from typing import List, Tuple, Set, Optional, Dict
import concurrent.futures

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

# Player Colors
COLOR_RED = (220, 60, 60)
COLOR_BLUE = (60, 100, 220)
COLOR_WALL_RED = (180, 40, 40)
COLOR_WALL_BLUE = (40, 80, 180)

class Player(Enum):
    RED = 0
    BLUE = 1

    def opposite(self):
        return Player.BLUE if self == Player.RED else Player.RED

class WallOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_WALL_PLACEMENTS = tuple(
    (orient, r, c)
    for orient in (WallOrientation.HORIZONTAL, WallOrientation.VERTICAL)
    for r in range(BOARD_SIZE - 1)
    for c in range(BOARD_SIZE - 1)
)

# --- Game Logic Implementation ---

class BarricadeState:
    def __init__(self):
        # Pawns: (row, col). RED starts at (0, 4) (top), BLUE at (8, 4) (bottom)
        self.pawns = {Player.RED: (0, 4), Player.BLUE: (8, 4)}
        # HORIZONTAL (r, c) blocks vertical movement across row r/r+1 at columns c and c+1.
        # VERTICAL (r, c) blocks horizontal movement across col c/c+1 at rows r and r+1.
        self.walls: Set[Tuple[WallOrientation, int, int]] = set()
        self.current_player = Player.RED
        self.winner = None
        self.walls_left = {Player.RED: 10, Player.BLUE: 10}

        self._valid_moves_cache_key = None
        self._valid_moves_cache = None
        self._path_cache = {}
        self._route_cache = {}

    def copy(self):
        new_state = BarricadeState.__new__(BarricadeState)
        new_state.pawns = dict(self.pawns)
        new_state.walls = set(self.walls)
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.walls_left = dict(self.walls_left)
        new_state._valid_moves_cache_key = None
        new_state._valid_moves_cache = None
        new_state._path_cache = self._path_cache
        new_state._route_cache = self._route_cache
        return new_state

    def _state_cache_key(self) -> Tuple:
        return (
            self.pawns[Player.RED],
            self.pawns[Player.BLUE],
            self.current_player,
            self.winner,
            self.walls_left[Player.RED],
            self.walls_left[Player.BLUE],
            frozenset(self.walls),
        )

    def get_valid_moves(self) -> List[Tuple]:
        """Returns list of valid moves: ('move', r, c) or ('wall', orientation, r, c)"""
        if self.winner is not None:
            return []

        key = self._state_cache_key()
        if key == self._valid_moves_cache_key and self._valid_moves_cache is not None:
            return list(self._valid_moves_cache)

        moves = self.get_pawn_moves()
        moves.extend(self.get_valid_wall_moves())
        self._valid_moves_cache_key = key
        self._valid_moves_cache = tuple(moves)
        return moves

    def get_pawn_moves(self) -> List[Tuple]:
        if self.winner is not None:
            return []

        moves = []
        pr, pc = self.pawns[self.current_player]
        op = self.current_player.opposite()
        or_, oc = self.pawns[op]

        for dr, dc in DIRECTIONS:
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

        return moves

    def get_valid_wall_moves(self, candidates=None) -> List[Tuple]:
        if self.walls_left[self.current_player] <= 0:
            return []

        wall_moves = []
        placements = ALL_WALL_PLACEMENTS if candidates is None else candidates
        for orient, r, c in placements:
            if self._is_valid_wall_placement(orient, r, c):
                wall_moves.append(('wall', orient, r, c))

        return wall_moves

    def _is_blocked(self, r1, c1, r2, c2) -> bool:
        """Check if a direct step between two adjacent cells is blocked by a wall."""
        if r1 == r2: # Horizontal move
            c_min = min(c1, c2)
            for wr in [r1, r1 - 1]:
                if 0 <= wr < BOARD_SIZE - 1 and (WallOrientation.VERTICAL, wr, c_min) in self.walls:
                    return True
        elif c1 == c2: # Vertical move
            r_min = min(r1, r2)
            for wc in [c1, c1 - 1]:
                if 0 <= wc < BOARD_SIZE - 1 and (WallOrientation.HORIZONTAL, r_min, wc) in self.walls:
                    return True
        return False

    def _is_valid_wall_placement(self, orient: WallOrientation, r: int, c: int) -> bool:
        if not self._is_wall_shape_available(orient, r, c):
            return False

        self.walls.add((orient, r, c))
        can_reach_red = self._has_path(Player.RED)
        can_reach_blue = self._has_path(Player.BLUE)
        self.walls.remove((orient, r, c))

        return can_reach_red and can_reach_blue

    def _is_wall_shape_available(self, orient: WallOrientation, r: int, c: int) -> bool:
        if r < 0 or r >= BOARD_SIZE - 1 or c < 0 or c >= BOARD_SIZE - 1:
            return False

        if orient == WallOrientation.HORIZONTAL:
            if (WallOrientation.HORIZONTAL, r, c) in self.walls: return False
            if (WallOrientation.HORIZONTAL, r, c - 1) in self.walls: return False
            if (WallOrientation.HORIZONTAL, r, c + 1) in self.walls: return False
            if (WallOrientation.VERTICAL, r, c) in self.walls: return False
        else: # Vertical
            if (WallOrientation.VERTICAL, r, c) in self.walls: return False
            if (WallOrientation.VERTICAL, r - 1, c) in self.walls: return False
            if (WallOrientation.VERTICAL, r + 1, c) in self.walls: return False
            if (WallOrientation.HORIZONTAL, r, c) in self.walls: return False

        return True

    def _has_path(self, player: Player) -> bool:
        return self.shortest_path_length(player) is not None

    def shortest_path_length(self, player: Player) -> Optional[int]:
        key = ('distance', player, self.pawns[player], frozenset(self.walls))
        if key in self._path_cache:
            return self._path_cache[key]

        start = self.pawns[player]
        target_row = 8 if player == Player.RED else 0

        queue = deque([(start, 0)])
        visited = set([start])

        while queue:
            (r, c), dist = queue.popleft()
            if r == target_row:
                self._path_cache[key] = dist
                return dist

            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) not in visited:
                        if not self._is_blocked(r, c, nr, nc):
                            visited.add((nr, nc))
                            queue.append(((nr, nc), dist + 1))

        self._path_cache[key] = None
        return None

    def shortest_path_cells(self, player: Player) -> Optional[Tuple[Tuple[int, int], ...]]:
        key = ('route', player, self.pawns[player], frozenset(self.walls))
        if key in self._route_cache:
            return self._route_cache[key]

        start = self.pawns[player]
        target_row = 8 if player == Player.RED else 0
        queue = deque([start])
        parent = {start: None}

        while queue:
            r, c = queue.popleft()
            if r == target_row:
                path = []
                node = (r, c)
                while node is not None:
                    path.append(node)
                    node = parent[node]
                result = tuple(reversed(path))
                self._route_cache[key] = result
                return result

            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and nxt not in parent:
                    if not self._is_blocked(r, c, nr, nc):
                        parent[nxt] = (r, c)
                        queue.append(nxt)

        self._route_cache[key] = None
        return None

    def apply_move(self, move: Tuple) -> 'BarricadeState':
        new_state = self.copy()
        type_ = move[0]

        if type_ == 'move':
            _, r, c = move
            new_state.pawns[new_state.current_player] = (r, c)
            # Check win
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

# --- AI Implementation ---

class BarricadeAI:
    def __init__(self, player: Player, max_depth: int = 3, max_wall_moves: int = 24):
        self.player = player
        self.max_depth = max_depth
        self.max_wall_moves = max_wall_moves
        self.tt = {}
        self._search_moves_cache = {}

    def get_best_move(self, state: BarricadeState) -> Optional[Tuple]:
        self.tt.clear()
        self._search_moves_cache.clear()
        moves = self._ordered_moves(state, self.max_depth)
        if not moves:
            return None

        best_score = -math.inf
        best_move = moves[0]
        alpha = -math.inf
        beta = math.inf

        for move in moves:
            score = self._minimax(state.apply_move(move), self.max_depth - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_move

    def _state_hash(self, state: BarricadeState) -> tuple:
        return state._state_cache_key()

    def _minimax(self, state: BarricadeState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        s_hash = self._state_hash(state)
        if s_hash in self.tt and self.tt[s_hash][1] >= depth:
            return self.tt[s_hash][0]

        if state.winner is not None:
            if state.winner == self.player:
                score = 1000 + depth # Prefer faster wins
            else:
                score = -1000 - depth
            self.tt[s_hash] = (score, depth)
            return score

        if depth == 0:
            score = self._evaluate(state)
            self.tt[s_hash] = (score, depth)
            return score

        moves = self._ordered_moves(state, depth)
        if not moves:
            moves = state.get_valid_moves()
        if not moves:
            score = -1000
            self.tt[s_hash] = (score, depth)
            return score

        if is_maximizing:
            max_eval = -math.inf
            completed = True
            for move in moves:
                eval_score = self._minimax(state.apply_move(move), depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    completed = False
                    break
            if completed:
                self.tt[s_hash] = (max_eval, depth)
            return max_eval
        else:
            min_eval = math.inf
            completed = True
            for move in moves:
                eval_score = self._minimax(state.apply_move(move), depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    completed = False
                    break
            if completed:
                self.tt[s_hash] = (min_eval, depth)
            return min_eval

    def _ordered_moves(self, state: BarricadeState, depth: int) -> List[Tuple]:
        cache_key = (state._state_cache_key(), depth)
        if cache_key in self._search_moves_cache:
            return list(self._search_moves_cache[cache_key])

        moves = state.get_pawn_moves()
        if state.walls_left[state.current_player] > 0:
            candidates = self._candidate_wall_placements(state)
            wall_moves = state.get_valid_wall_moves(candidates)
            wall_moves.sort(key=lambda move: self._move_order_score(state, move), reverse=True)
            wall_limit = self.max_wall_moves if depth >= self.max_depth - 1 else max(8, self.max_wall_moves // 2)
            moves.extend(wall_moves[:wall_limit])

        moves.sort(key=lambda move: self._move_order_score(state, move), reverse=True)
        self._search_moves_cache[cache_key] = tuple(moves)
        return moves

    def _candidate_wall_placements(self, state: BarricadeState) -> List[Tuple[WallOrientation, int, int]]:
        focus_cells = set(state.pawns.values())
        for player in (Player.RED, Player.BLUE):
            path = state.shortest_path_cells(player)
            if path:
                focus_cells.update(path)

        candidates = set()
        for row, col in focus_cells:
            for r in range(row - 1, row + 2):
                for c in range(col - 1, col + 2):
                    if 0 <= r < BOARD_SIZE - 1 and 0 <= c < BOARD_SIZE - 1:
                        candidates.add((WallOrientation.HORIZONTAL, r, c))
                        candidates.add((WallOrientation.VERTICAL, r, c))

        return sorted(candidates, key=lambda item: (item[0].value, item[1], item[2]))

    def _move_order_score(self, state: BarricadeState, move: Tuple) -> float:
        current = state.current_player
        if move[0] == 'move':
            _, r, c = move
            progress = r if current == Player.RED else (BOARD_SIZE - 1 - r)
            center_bonus = BOARD_SIZE // 2 - abs(c - BOARD_SIZE // 2)
            return 100 + progress * 4 + center_bonus

        _, _, r, c = move
        opponent = current.opposite()
        op_r, op_c = state.pawns[opponent]
        distance_to_opponent = abs(r - op_r) + abs(c - op_c)
        return 60 - distance_to_opponent * 3

    def _evaluate(self, state: BarricadeState) -> float:
        # Heuristic: Distance to goal - Opponent distance to goal + Wall advantage
        dist_me = self._shortest_path(state, self.player)
        dist_opp = self._shortest_path(state, self.player.opposite())

        if dist_me is None: dist_me = 100 # Should not happen due to validation
        if dist_opp is None: dist_opp = 100

        score = (dist_opp - dist_me) * 10

        # Small bonus for wall count
        if hasattr(state, 'walls_left'):
            score += (state.walls_left[self.player] - state.walls_left[self.player.opposite()]) * 2

        my_pos = state.pawns[self.player]
        opp_pos = state.pawns[self.player.opposite()]
        score += (4 - abs(my_pos[1] - 4)) * 0.5
        score -= (4 - abs(opp_pos[1] - 4)) * 0.5

        return score

    def _shortest_path(self, state: BarricadeState, player: Player) -> Optional[int]:
        return state.shortest_path_length(player)

# --- Pygame UI Implementation ---

class BarricadeUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Barricade AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.big_font = pygame.font.SysFont("Arial", 24, bold=True)

        self.state = BarricadeState()
        self.ai = BarricadeAI(Player.BLUE, max_depth=3) # AI plays Blue
        self.engine_starts = False

        self.mode = 'move' # 'move' or 'wall'
        self.wall_orientation = WallOrientation.HORIZONTAL

        self.hover_pos = None # (row, col) for cell hover
        self.hover_wall_pos = None # (orient, r, c) for wall hover
        self.valid_moves_cache = []
        self.valid_moves_set = set()

        self.game_over = False
        self.ai_thinking = False
        self.last_click_time = 0
        
        self.ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.ai_future = None
        self.reset_game()

    def refresh_valid_moves(self):
        self.valid_moves_cache = self.state.get_valid_moves()
        self.valid_moves_set = set(self.valid_moves_cache)

    def reset_game(self):
        self.state = BarricadeState()
        if self.engine_starts:
            self.state.current_player = Player.BLUE

        self.game_over = False
        self.ai_thinking = self.engine_starts
        self.ai_future = None
        self.message = "AI Thinking..." if self.engine_starts else "Your Turn (Red)"
        self.refresh_valid_moves()

    def toggle_first_move(self):
        self.engine_starts = not self.engine_starts
        self.reset_game()

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
                if self.ai_future is None:
                    self.ai_future = self.ai_executor.submit(self.ai.get_best_move, self.state.copy())
                elif self.ai_future.done():
                    best_move = self.ai_future.result()
                    if best_move:
                        self.state = self.state.apply_move(best_move)
                        self.check_game_status()
                        if not self.game_over:
                            self.message = "Your Turn (Red)"
                            self.refresh_valid_moves()
                    self.ai_thinking = False
                    self.ai_future = None

        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if self.game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_f:
                        self.toggle_first_move()
                continue

            if self.ai_thinking:
                continue

            if event.type == pygame.MOUSEMOTION:
                pass # Handled in update_hover

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_w:
                    self.mode = 'wall' if self.mode == 'move' else 'move'
                    # Refresh cache when mode changes to show correct highlights
                    self.refresh_valid_moves()
                if event.key == pygame.K_r:
                    self.wall_orientation = WallOrientation.VERTICAL if self.wall_orientation == WallOrientation.HORIZONTAL else WallOrientation.HORIZONTAL
                if event.key == pygame.K_f:
                    self.toggle_first_move()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    self.handle_click()
                elif event.button == 3: # Right Click
                    self.wall_orientation = WallOrientation.VERTICAL if self.wall_orientation == WallOrientation.HORIZONTAL else WallOrientation.HORIZONTAL

    def get_grid_pos_from_mouse(self):
        mx, my = pygame.mouse.get_pos()
        # Adjust for margin
        gx = mx - MARGIN
        gy = my - MARGIN

        if 0 <= gx < BOARD_SIZE * CELL_SIZE and 0 <= gy < BOARD_SIZE * CELL_SIZE:
            col = gx // CELL_SIZE
            row = gy // CELL_SIZE
            return row, col
        return None

    def update_hover(self):
        mpos = pygame.mouse.get_pos()
        # Calculate raw grid float coordinates for precise wall hovering
        mx, my = mpos
        gx = mx - MARGIN
        gy = my - MARGIN

        self.hover_pos = None
        self.hover_wall_pos = None

        if 0 <= gx < BOARD_SIZE * CELL_SIZE and 0 <= gy < BOARD_SIZE * CELL_SIZE:
            col = gx / CELL_SIZE
            row = gy / CELL_SIZE

            # Cell hover
            c_col = int(col)
            c_row = int(row)
            self.hover_pos = (c_row, c_col)

            # Wall hover logic
            # Determine which intersection we are near
            # Intersections are at integer boundaries of cells.
            # Cell (r, c) is from x=c to c+1, y=r to r+1.
            # Intersections: (r, c) is top-left of cell (r,c).
            # We need to map mouse to the nearest valid wall anchor.

            # Relative position inside the cell (0.0 to 1.0)
            rx = col - c_col
            ry = row - c_row

            # Determine closest grid line intersection
            # If rx < 0.5, closer to left line (c_col), else right (c_col+1)
            # If ry < 0.5, closer to top line (c_row), else bottom (c_row+1)

            near_x = c_col if rx < 0.5 else c_col + 1
            near_y = c_row if ry < 0.5 else c_row + 1

            # Now determine orientation based on proximity to lines
            # Distance to vertical line vs horizontal line
            dist_v = abs(rx - 0.5) # 0 at center, 0.5 at edges? No.
            # Let's simplify: Use the quadrant to decide intended orientation if not forced?
            # Or just check both possible walls near this intersection and see which one the mouse is "along".

            # Actually, easier: The user toggles orientation with R.
            # We just highlight the wall of the CURRENT orientation nearest to the mouse.

            # Anchor points for walls:
            # H wall (r, c) is centered at (r+0.5, c+0.5) visually? No, it spans (c, c+1) at row boundary r/r+1.
            # Its "clickable" area is roughly the gap between cells.

            # Let's define specific hover zones.
            # If Orientation is HORIZONTAL:
            # We look for the horizontal gap.
            # Mouse Y determines the row gap. Mouse X determines the column span.
            # Row gap index: int(row) if row%1 < 0.5 else int(row)+1 ?
            # Let's use the nearest intersection approach.

            # Candidate H wall: row_boundary = round(row) - 1?
            # If mouse is at y=10.2 (cell 10), top boundary is 10, bottom is 11.
            # If ry < 0.5, closer to top (c_row). Wall row = c_row-1?
            # Wait, H wall at (r, c) is BETWEEN row r and r+1.
            # So if we are near the line y=c_row, the wall row is c_row-1 (if c_row>0) or c_row (if c_row<9)?
            # Actually, if we are in cell (r, c), the top edge corresponds to H wall at (r-1, c) or (r-1, c-1).
            # The bottom edge corresponds to H wall at (r, c) or (r, c-1).

            # Let's just calculate the nearest valid wall coordinate for the current orientation.
            if self.mode == 'wall':
                # Determine potential wall coords based on mouse float pos
                # H Wall: needs row_idx (0..7) and col_idx (0..7)
                # row_idx corresponds to the gap between row_idx and row_idx+1.
                # Closest gap index = int(round(row - 0.5))?
                # If row=0.2 -> -0.3 -> 0. Gap 0.
                # If row=0.8 -> 0.3 -> 0. Gap 0.
                # If row=1.2 -> 0.7 -> 1. Gap 1.
                h_r = int(round(row - 0.5))
                h_c = int(col) # Spanning this column index?
                # H wall at (r, c) spans c and c+1. So if mouse is at 0.2 (col 0), it's wall col 0.
                # If mouse at 0.8 (col 0), still wall col 0? Or col -1?
                # Center of wall (r, c) is at x = c + 0.5.
                # So h_c = int(round(col - 0.5))
                h_c = int(round(col - 0.5))

                # V Wall
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
        # Debounce to prevent double-click issues
        if current_time - self.last_click_time < 0.1:
            return
        self.last_click_time = current_time

        if self.mode == 'move' and self.hover_pos:
            r, c = self.hover_pos
            # Check if valid move
            target_move = ('move', r, c)
            if target_move in self.valid_moves_set:
                self.state = self.state.apply_move(target_move)
                self.check_game_status()
                if not self.game_over:
                    self.ai_thinking = True
                    self.message = "AI Thinking..."
                # Update cache for next frame
                self.refresh_valid_moves()

        elif self.mode == 'wall' and self.hover_wall_pos:
            orient, r, c = self.hover_wall_pos
            target_move = ('wall', orient, r, c)
            if target_move in self.valid_moves_set:
                self.state = self.state.apply_move(target_move)
                self.check_game_status()
                if not self.game_over:
                    self.ai_thinking = True
                    self.message = "AI Thinking..."
                # Update cache for next frame
                self.refresh_valid_moves()

    def check_game_status(self):
        if self.state.winner:
            self.game_over = True
            if self.state.winner == Player.RED:
                self.message = "You Win! Press 'R' to restart."
            else:
                self.message = "AI Wins! Press 'R' to restart."
        else:
            self.refresh_valid_moves()
            if not self.valid_moves_cache:
                # No moves available? Loss for current player?
                # In this game, if you can't move, you lose? Or just skip?
                # Assuming loss for simplicity if no moves and not won.
                self.game_over = True
                self.message = f"No moves for {self.state.current_player.name}. Game Over."

    def draw(self):
        self.screen.fill(COLOR_BG)

        # Draw Info Panel
        panel_x = MARGIN * 2 + BOARD_SIZE * CELL_SIZE
        pygame.draw.rect(self.screen, (40, 40, 50), (panel_x - 10, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))

        info_y = 20
        title = self.big_font.render("BARRICADE", True, COLOR_TEXT)
        self.screen.blit(title, (panel_x, info_y))
        info_y += 40

        # Status
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

        # Mode
        mode_txt = f"Mode: {self.mode.upper()}"
        color_mode = (100, 255, 100) if self.mode == 'move' else (255, 200, 100)
        txt = self.font.render(mode_txt, True, color_mode)
        self.screen.blit(txt, (panel_x, info_y))
        info_y += 25

        # Orientation
        if self.mode == 'wall':
            orient_txt = f"Orientation: {self.wall_orientation.name}"
            txt = self.font.render(orient_txt, True, COLOR_TEXT)
            self.screen.blit(txt, (panel_x, info_y))
            info_y += 25

        # Walls left display
        info_y += 10
        first_move = "Engine" if self.engine_starts else "Human"
        txt = self.font.render(f"First Move: {first_move}", True, COLOR_TEXT)
        self.screen.blit(txt, (panel_x, info_y))
        info_y += 25

        walls_red = self.state.walls_left[Player.RED] if hasattr(self.state, 'walls_left') else 10
        walls_blue = self.state.walls_left[Player.BLUE] if hasattr(self.state, 'walls_left') else 10
        txt = self.font.render(f"Red Walls: {walls_red}", True, COLOR_RED)
        self.screen.blit(txt, (panel_x, info_y))
        info_y += 25
        txt = self.font.render(f"Blue Walls: {walls_blue}", True, COLOR_BLUE)
        self.screen.blit(txt, (panel_x, info_y))
        info_y += 25

        # Controls
        info_y += 20
        controls = [
            "Controls:",
            "- L-Click: Action",
            "- R-Click: Rotate Wall",
            "- 'W': Toggle Mode",
            "- 'R': Rotate Wall",
            "- 'F': Toggle First",
            "- ESC: Quit"
        ]
        for line in controls:
            txt = self.font.render(line, True, (180, 180, 180))
            self.screen.blit(txt, (panel_x, info_y))
            info_y += 20

        # Message
        msg_surf = self.font.render(self.message, True, (255, 255, 255))
        self.screen.blit(msg_surf, (panel_x, WINDOW_HEIGHT - 50))

        # Draw Board Grid
        board_rect = pygame.Rect(MARGIN, MARGIN, BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE)
        pygame.draw.rect(self.screen, COLOR_GRID, board_rect, 2)

        # Draw Cells (optional subtle background)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = pygame.Rect(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Checkerboard pattern?
                if (r + c) % 2 == 0:
                    pygame.draw.rect(self.screen, (35, 35, 40), rect)
                else:
                    pygame.draw.rect(self.screen, (30, 30, 35), rect)

        # Draw Valid Move Highlights (for Move Mode)
        if self.mode == 'move' and self.hover_pos and not self.ai_thinking:
            r, c = self.hover_pos
            rect = pygame.Rect(MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if ('move', r, c) in self.valid_moves_set:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, COLOR_HIGHLIGHT_VALID, (0, 0, CELL_SIZE, CELL_SIZE))
                self.screen.blit(s, rect.topleft)
            else:
                # Show invalid if hovering over non-move
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, COLOR_HIGHLIGHT_INVALID, (0, 0, CELL_SIZE, CELL_SIZE), 3)
                self.screen.blit(s, rect.topleft)

        # Draw Walls
        for w in self.state.walls:
            orient, r, c = w
            color = COLOR_WALL_RED if self.state.current_player.opposite() == Player.RED else COLOR_WALL_BLUE
            # Determine who placed it? State doesn't track history easily without extra logic.
            # Let's just color by orientation or default.
            # Better: Check wall count difference? No.
            # Just use a neutral wall color or alternate. Let's use Yellow-ish for visibility.
            color = (200, 200, 50)

            thickness = 8
            if orient == WallOrientation.HORIZONTAL:
                # Between row r and r+1, from col c to c+1
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + (r + 1) * CELL_SIZE - thickness // 2
                w_len = CELL_SIZE * 2
                h_len = thickness
            else:
                # Between col c and c+1, from row r to r+1
                x = MARGIN + (c + 1) * CELL_SIZE - thickness // 2
                y = MARGIN + r * CELL_SIZE
                w_len = thickness
                h_len = CELL_SIZE * 2

            pygame.draw.rect(self.screen, color, (x, y, w_len, h_len))

        # Draw Wall Preview (Transparency)
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
            if move_tuple in self.valid_moves_set:
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

            # Label
            label = "R" if player == Player.RED else "B"
            txt = self.big_font.render(label, True, (255, 255, 255))
            text_rect = txt.get_rect(center=(center_x, center_y))
            self.screen.blit(txt, text_rect)

if __name__ == "__main__":
    game = BarricadeUI()
    game.run()
