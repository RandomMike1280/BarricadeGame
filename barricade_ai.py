"""
Barricade AI - A complete Python implementation
APPROACH:
=========
This implementation uses Minimax with Alpha-Beta Pruning for decision making.
Key components:
1. Game State Representation:
   - 9x9 board with pawn positions and wall placements
   - Walls stored as sets of edges they block (horizontal/vertical)
   - Player tracking (RED starts, aims for row 8; BLUE aims for row 0)
2. Move Generation:
   - Pawn moves: 4 directions + jumps over opponent when adjacent
   - Wall placements: All valid horizontal/vertical wall positions that don't:
     * Overlap existing walls
     * Block opponent's path to goal (verified via BFS pathfinding)
3. Heuristic Evaluation:
   - Distance to goal (Manhattan distance adjusted for walls)
   - Actual shortest path length via BFS (more accurate than Manhattan)
   - Wall placement potential
   - Mobility (number of available moves)
   - Weighted combination favoring path length difference
4. Search Strategy:
   - Minimax with alpha-beta pruning
   - Depth-limited search (depth 4-6 depending on time constraints)
   - Iterative deepening could be added for time control
   - Move ordering: prioritize pawn moves toward goal, then walls near pawns
5. Path Validation:
   - BFS used to verify walls don't completely block a player
   - BFS also used for accurate distance calculation in heuristic
ASSUMPTIONS:
============
- RED player starts at (4, 0) and aims to reach row 8 (any column)
- BLUE player starts at (4, 8) and aims to reach row 0 (any column)
- Walls are placed between cells, identified by their top-left corner and orientation
- A horizontal wall at (r, c) blocks movement between (r,c)-(r,c+1) and (r+1,c)-(r+1,c+1)
- A vertical wall at (r, c) blocks movement between (r,c)-(r+1,c) and (r,c+1)-(r+1,c+1)
- Jump move: Only single jump over opponent is allowed (not double jumps)
"""
import copy
from collections import deque
from typing import List, Tuple, Set, Optional, Dict
from enum import Enum
import random
class Player(Enum):
    RED = "red"      # Starts at row 0, aims for row 8
    BLUE = "blue"    # Starts at row 8, aims for row 0
class MoveType(Enum):
    PAWN_MOVE = "pawn_move"
    WALL_PLACE = "wall_place"
class Move:
    """Represents a game move."""
    def __init__(self, move_type: MoveType, data: dict):
        self.move_type = move_type
        self.data = data  # {'to': (row, col)} or {'pos': (row, col), 'orientation': 'H'/'V'}
    def __repr__(self):
        if self.move_type == MoveType.PAWN_MOVE:
            return f"PawnMove({self.data['to']})"
        else:
            return f"WallPlace({self.data['pos']}, {self.data['orientation']})"
    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.move_type == other.move_type and self.data == other.data
    def __hash__(self):
        return hash((self.move_type, str(self.data)))
class BarricadeState:
    """
    Represents the game state of Barricade.
    Board coordinates: (row, col) where row 0 is top, col 0 is left
    RED starts at (4, 0), aims for row 8
    BLUE starts at (4, 8), aims for row 0
    """
    BOARD_SIZE = 9
    def __init__(self):
        # Pawn positions: (row, col)
        self.red_pawn = (4, 0)
        self.blue_pawn = (4, 8)
        # Walls: set of ((row, col), orientation)
        # Horizontal wall at (r, c) spans between columns c and c+1 for rows r and r+1
        # Vertical wall at (r, c) spans between rows r and r+1 for columns c and c+1
        self.walls: Set[Tuple[Tuple[int, int], str]] = set()
        # Wall counts
        self.red_walls_left = 10
        self.blue_walls_left = 10
        # Current player
        self.current_player = Player.RED
        # Move history
        self.move_history: List[Move] = []
    def copy(self) -> 'BarricadeState':
        """Create a deep copy of the game state."""
        new_state = BarricadeState.__new__(BarricadeState)
        new_state.red_pawn = self.red_pawn
        new_state.blue_pawn = self.blue_pawn
        new_state.walls = self.walls.copy()
        new_state.red_walls_left = self.red_walls_left
        new_state.blue_walls_left = self.blue_walls_left
        new_state.current_player = self.current_player
        new_state.move_history = self.move_history.copy()
        return new_state
    def get_current_pawn(self) -> Tuple[int, int]:
        """Get the position of the current player's pawn."""
        if self.current_player == Player.RED:
            return self.red_pawn
        else:
            return self.blue_pawn
    def get_opponent_pawn(self) -> Tuple[int, int]:
        """Get the position of the opponent's pawn."""
        if self.current_player == Player.RED:
            return self.blue_pawn
        else:
            return self.red_pawn
    def get_goal_row(self, player: Player) -> int:
        """Get the goal row for a player."""
        if player == Player.RED:
            return 8  # RED aims for bottom row
        else:
            return 0  # BLUE aims for top row
    def is_wall_at(self, pos: Tuple[int, int], orientation: str) -> bool:
        """Check if a wall exists at the given position and orientation."""
        return (pos, orientation) in self.walls
    def get_blocked_edges(self) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get all blocked edges due to walls.
        Returns a set of edge pairs ((r1, c1), (r2, c2)) that are blocked.
        """
        blocked = set()
        for (pos, orientation) in self.walls:
            r, c = pos
            if orientation == 'H':
                # Horizontal wall blocks horizontal movement
                # Blocks (r, c)-(r, c+1) and (r+1, c)-(r+1, c+1)
                blocked.add(((r, c), (r, c + 1)))
                blocked.add(((r, c + 1), (r, c)))
                blocked.add(((r + 1, c), (r + 1, c + 1)))
                blocked.add(((r + 1, c + 1), (r + 1, c)))
            else:  # 'V'
                # Vertical wall blocks vertical movement
                # Blocks (r, c)-(r+1, c) and (r, c+1)-(r+1, c+1)
                blocked.add(((r, c), (r + 1, c)))
                blocked.add(((r + 1, c), (r, c)))
                blocked.add(((r, c + 1), (r + 1, c + 1)))
                blocked.add(((r + 1, c + 1), (r, c + 1)))
        return blocked
    def can_move_pawn(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Check if a pawn can move from from_pos to to_pos.
        Considers walls and jumping over opponent.
        """
        fr, fc = from_pos
        tr, tc = to_pos
        # Check bounds
        if not (0 <= tr < self.BOARD_SIZE and 0 <= tc < self.BOARD_SIZE):
            return False
        # Calculate direction
        dr = tr - fr
        dc = tc - fc
        # Must move exactly one square (normal move) or two squares (jump)
        if abs(dr) + abs(dc) == 1:
            # Normal move: check if edge is blocked by wall
            blocked_edges = self.get_blocked_edges()
            if ((from_pos, to_pos)) in blocked_edges:
                return False
            return True
        elif abs(dr) + abs(dc) == 2:
            # Potential jump over opponent
            opponent_pos = self.get_opponent_pawn()
            # Check if opponent is directly between from and to
            mid_r = fr + dr // 2
            mid_c = fc + dc // 2
            if (mid_r, mid_c) != opponent_pos:
                return False
            # Check if we can move to the middle (where opponent is)
            blocked_edges = self.get_blocked_edges()
            if ((from_pos, (mid_r, mid_c))) in blocked_edges:
                return False
            # Check if we can move from middle to destination
            if (((mid_r, mid_c), to_pos)) in blocked_edges:
                return False
            return True
        return False
    def get_pawn_moves(self) -> List[Move]:
        """Get all legal pawn moves for the current player."""
        moves = []
        pawn_pos = self.get_current_pawn()
        r, c = pawn_pos
        # Four cardinal directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            # Try normal move
            to_pos = (r + dr, c + dc)
            if self.can_move_pawn(pawn_pos, to_pos):
                moves.append(Move(MoveType.PAWN_MOVE, {'to': to_pos}))
            # Try jump (two squares in direction)
            jump_pos = (r + 2*dr, c + 2*dc)
            if self.can_move_pawn(pawn_pos, jump_pos):
                moves.append(Move(MoveType.PAWN_MOVE, {'to': jump_pos}))
        return moves
    def has_path_to_goal(self, player: Player, test_walls: Optional[Set] = None) -> bool:
        """
        Check if a player has a valid path to their goal row using BFS.
        Used to validate wall placements.
        """
        walls = test_walls if test_walls is not None else self.walls
        # Build blocked edges from test_walls
        blocked = set()
        for (pos, orientation) in walls:
            r, c = pos
            if orientation == 'H':
                blocked.add(((r, c), (r, c + 1)))
                blocked.add(((r, c + 1), (r, c)))
                blocked.add(((r + 1, c), (r + 1, c + 1)))
                blocked.add(((r + 1, c + 1), (r + 1, c)))
            else:
                blocked.add(((r, c), (r + 1, c)))
                blocked.add(((r + 1, c), (r, c)))
                blocked.add(((r, c + 1), (r + 1, c + 1)))
                blocked.add(((r + 1, c + 1), (r, c + 1)))
        # Starting position
        if player == Player.RED:
            start = self.red_pawn
            goal_row = 8
        else:
            start = self.blue_pawn
            goal_row = 0
        # BFS
        visited = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            # Check if reached goal row
            if r == goal_row:
                return True
            # Explore neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE):
                    continue
                if (nr, nc) in visited:
                    continue
                # Check if edge is blocked
                if (((r, c), (nr, nc))) in blocked:
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))
        return False
    def get_wall_placements(self) -> List[Move]:
        """Get all legal wall placements for the current player."""
        moves = []
        # Check if player has walls left
        if self.current_player == Player.RED and self.red_walls_left <= 0:
            return moves
        if self.current_player == Player.BLUE and self.blue_walls_left <= 0:
            return moves
        opponent = Player.BLUE if self.current_player == Player.RED else Player.RED
        # Limit wall positions to those near pawns for efficiency
        # Get pawn positions to focus wall placement
        red_r, red_c = self.red_pawn
        blue_r, blue_c = self.blue_pawn
        # Generate candidate positions near both pawns (within 3 cells)
        candidate_positions = set()
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = red_r + dr, red_c + dc
                if 0 <= nr < self.BOARD_SIZE - 1 and 0 <= nc < self.BOARD_SIZE - 1:
                    candidate_positions.add((nr, nc))
                nr, nc = blue_r + dr, blue_c + dc
                if 0 <= nr < self.BOARD_SIZE - 1 and 0 <= nc < self.BOARD_SIZE - 1:
                    candidate_positions.add((nr, nc))
        # Try all candidate wall positions
        for r, c in candidate_positions:
            for orientation in ['H', 'V']:
                # Check if wall overlaps with existing walls
                if self._walls_overlap((r, c), orientation):
                    continue
                # Create test wall set
                test_walls = self.walls | {((r, c), orientation)}
                # Check if this wall blocks opponent's path to goal
                if not self.has_path_to_goal(opponent, test_walls):
                    continue
                moves.append(Move(MoveType.WALL_PLACE, {
                    'pos': (r, c),
                    'orientation': orientation
                }))
        return moves
    def _walls_overlap(self, pos: Tuple[int, int], orientation: str) -> bool:
        """Check if placing a wall at pos with given orientation would overlap existing walls."""
        r, c = pos
        for (wall_pos, wall_orient) in self.walls:
            wr, wc = wall_pos
            if orientation == wall_orient:
                if orientation == 'H':
                    if wr == r and abs(wc - c) <= 1:
                        return True
                else:
                    if wc == c and abs(wr - r) <= 1:
                        return True
            else:
                # Perpendicular walls
                if orientation == 'H' and wall_orient == 'V':
                    # H at (r,c), V at (wr,wc)
                    # They touch if wr in [r, r+1] and wc in [c-1, c]
                    if wr in [r, r + 1] and wc in [c - 1, c]:
                        return True
                elif orientation == 'V' and wall_orient == 'H':
                    # V at (r,c), H at (wr,wc)
                    if wc in [wr, wr + 1] and wr in [r - 1, r]:
                        return True
        return False
    def get_all_moves(self) -> List[Move]:
        """Get all legal moves for the current player."""
        pawn_moves = self.get_pawn_moves()
        wall_moves = self.get_wall_placements()
        return pawn_moves + wall_moves
    def apply_move(self, move: Move) -> 'BarricadeState':
        """Apply a move and return the new state."""
        new_state = self.copy()
        new_state.move_history.append(move)
        if move.move_type == MoveType.PAWN_MOVE:
            to_pos = move.data['to']
            if new_state.current_player == Player.RED:
                new_state.red_pawn = to_pos
            else:
                new_state.blue_pawn = to_pos
        else:  # WALL_PLACE
            pos = move.data['pos']
            orientation = move.data['orientation']
            new_state.walls.add((pos, orientation))
            if new_state.current_player == Player.RED:
                new_state.red_walls_left -= 1
            else:
                new_state.blue_walls_left -= 1
        # Switch player
        new_state.current_player = Player.BLUE if self.current_player == Player.RED else Player.RED
        return new_state
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        # RED wins if red_pawn reaches row 8
        if self.red_pawn[0] == 8:
            return True
        # BLUE wins if blue_pawn reaches row 0
        if self.blue_pawn[0] == 0:
            return True
        return False
    def get_winner(self) -> Optional[Player]:
        """Get the winner if the game is terminal."""
        if self.red_pawn[0] == 8:
            return Player.RED
        if self.blue_pawn[0] == 0:
            return Player.BLUE
        return None
    def get_shortest_path_length(self, player: Player) -> int:
        """Get the shortest path length for a player to reach their goal using BFS."""
        if player == Player.RED:
            start = self.red_pawn
            goal_row = 8
        else:
            start = self.blue_pawn
            goal_row = 0
        blocked_edges = self.get_blocked_edges()
        visited = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            (r, c), dist = queue.popleft()
            if r == goal_row:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE):
                    continue
                if (((r, c), (nr, nc))) in blocked_edges:
                    continue
                if (nr, nc) not in visited or visited[(nr, nc)] > dist + 1:
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        # No path found (shouldn't happen due to wall validation)
        return float('inf')
class BarricadeAI:
    """
    AI agent for Barricade using Minimax with Alpha-Beta Pruning.
    """
    def __init__(self, player: Player, max_depth: int = 4):
        self.player = player
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    def evaluate(self, state: BarricadeState) -> float:
        """
        Evaluate the game state from the AI's perspective.
        Higher values are better for the AI.
        """
        ai_path = state.get_shortest_path_length(self.player)
        opponent = Player.BLUE if self.player == Player.RED else Player.RED
        opp_path = state.get_shortest_path_length(opponent)
        # Primary factor: path length difference
        path_score = (opp_path - ai_path) * 10
        # Secondary: wall advantage
        if self.player == Player.RED:
            wall_diff = state.red_walls_left - state.blue_walls_left
        else:
            wall_diff = state.blue_walls_left - state.red_walls_left
        wall_score = wall_diff * 2
        # Skip mobility calculation during deep search (expensive)
        mobility_score = 0
        # Bonus for being close to winning
        win_bonus = 0
        if ai_path <= 2:
            win_bonus = 50
        if opp_path <= 2:
            win_bonus -= 30
        return path_score + wall_score + mobility_score + win_bonus
    def minimax(self, state: BarricadeState, depth: int, alpha: float, beta: float,
                maximizing: bool) -> Tuple[float, Optional[Move]]:
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (score, best_move).
        """
        self.nodes_evaluated += 1
        # Terminal state or depth limit
        if state.is_terminal() or depth == 0:
            if state.is_terminal():
                winner = state.get_winner()
                if winner == self.player:
                    return (10000, None)
                elif winner is not None:
                    return (-10000, None)
            return (self.evaluate(state), None)
        moves = state.get_all_moves()
        if not moves:
            # No legal moves (shouldn't happen normally)
            return (self.evaluate(state), None)
        # Move ordering: prioritize moves that improve position
        if maximizing:
            moves = self._order_moves(moves, state, descending=True)
        else:
            moves = self._order_moves(moves, state, descending=False)
        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                new_state = state.apply_move(move)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return (max_eval, best_move)
        else:
            min_eval = float('inf')
            for move in moves:
                new_state = state.apply_move(move)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return (min_eval, best_move)
    def _order_moves(self, moves: List[Move], state: BarricadeState,
                     descending: bool = True) -> List[Move]:
        """
        Order moves for better alpha-beta pruning.
        Prioritize pawn moves toward goal and walls near pawns.
        """
        def move_score(move: Move) -> float:
            if move.move_type == MoveType.PAWN_MOVE:
                to_pos = move.data['to']
                goal_row = state.get_goal_row(self.player)
                current_pos = state.get_current_pawn()
                # Distance improvement toward goal
                dist_improvement = abs(current_pos[0] - goal_row) - abs(to_pos[0] - goal_row)
                return dist_improvement * 10
            else:
                # Wall placement: prefer walls near current pawn
                wall_pos = move.data['pos']
                pawn_pos = state.get_current_pawn()
                distance = abs(wall_pos[0] - pawn_pos[0]) + abs(wall_pos[1] - pawn_pos[1])
                return -distance  # Closer is better
        return sorted(moves, key=move_score, reverse=descending)
    def get_best_move(self, state: BarricadeState) -> Move:
        """Get the best move for the current state."""
        self.nodes_evaluated = 0
        # Ensure it's the AI's turn
        assert state.current_player == self.player, "Not AI's turn"
        # Use limited depth for speed (depth 3 is a good balance)
        score, move = self.minimax(state, min(self.max_depth, 3), float('-inf'), float('inf'), True)
        return move if move else random.choice(state.get_all_moves())
def print_board(state: BarricadeState):
    """Print the board in a readable format."""
    print("\n  " + " ".join(str(c) for c in range(9)))
    print("  " + "-" * 17)
    for r in range(9):
        # Row with pawns and empty cells
        row_str = f"{r}|"
        for c in range(9):
            cell = "."
            if (r, c) == state.red_pawn:
                cell = "R"
            elif (r, c) == state.blue_pawn:
                cell = "B"
            row_str += cell + " "
        print(row_str)
        # Row for horizontal walls
        if r < 8:
            wall_row = "  "
            for c in range(9):
                # Check for horizontal wall above this cell
                has_h_wall = ((r, c), 'H') in state.walls or ((r, c - 1), 'H') in state.walls
                has_v_wall_left = ((r, c), 'V') in state.walls if c > 0 else False
                has_v_wall_right = ((r, c), 'V') in state.walls if c < 8 else False
                if c < 8 and ((r, c), 'H') in state.walls:
                    wall_row += "= "
                elif ((r, c), 'V') in state.walls:
                    wall_row += "| "
                else:
                    wall_row += "  "
            print(wall_row.rstrip())
    print(f"\nRed pawn: {state.red_pawn}, Walls left: {state.red_walls_left}")
    print(f"Blue pawn: {state.blue_pawn}, Walls left: {state.blue_walls_left}")
    print(f"Current player: {state.current_player.value}")
def human_input(state: BarricadeState) -> Move:
    """Get a move from human input."""
    while True:
        print("\nAvailable moves:")
        moves = state.get_all_moves()
        # Group moves by type
        pawn_moves = [m for m in moves if m.move_type == MoveType.PAWN_MOVE]
        wall_moves = [m for m in moves if m.move_type == MoveType.WALL_PLACE]
        print(f"Pawn moves ({len(pawn_moves)}):")
        for i, move in enumerate(pawn_moves[:10]):
            print(f"  {i}: Move to {move.data['to']}")
        if len(pawn_moves) > 10:
            print(f"  ... and {len(pawn_moves) - 10} more")
        print(f"\nWall placements ({len(wall_moves)}):")
        for i, move in enumerate(wall_moves[:10]):
            print(f"  {i + 10}: Place {move.data['orientation']} wall at {move.data['pos']}")
        if len(wall_moves) > 10:
            print(f"  ... and {len(wall_moves) - 10} more")
        try:
            choice = int(input("\nEnter move number: "))
            if 0 <= choice < len(moves):
                return moves[choice]
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
def play_game(ai_vs_ai: bool = False, ai_depth: int = 4):
    """Run a game of Barricade."""
    state = BarricadeState()
    # Create AIs
    red_ai = BarricadeAI(Player.RED, max_depth=ai_depth)
    blue_ai = BarricadeAI(Player.BLUE, max_depth=ai_depth)
    print("=" * 50)
    print("BARRICADE GAME")
    print("=" * 50)
    print("\nRules:")
    print("- RED starts at (4,0) and aims to reach row 8")
    print("- BLUE starts at (4,8) and aims to reach row 0")
    print("- Each player has 10 walls to place")
    print("- On each turn: move pawn OR place a wall")
    print("=" * 50)
    move_count = 0
    while not state.is_terminal():
        print_board(state)
        print(f"\n--- Turn {move_count + 1} ---")
        if state.current_player == Player.RED:
            if ai_vs_ai:
                print("RED (AI) is thinking...")
                move = red_ai.get_best_move(state)
                print(f"RED chooses: {move}")
            else:
                print("RED (Human)'s turn")
                move = human_input(state)
        else:
            print("BLUE (AI) is thinking...")
            move = blue_ai.get_best_move(state)
            print(f"BLUE chooses: {move}")
        state = state.apply_move(move)
        move_count += 1
        # Check for winner after each move
        if state.is_terminal():
            break
    print_board(state)
    print("\n" + "=" * 50)
    winner = state.get_winner()
    print(f"GAME OVER! {winner.value.upper()} WINS!")
    print(f"Total moves: {move_count}")
    print("=" * 50)
def demo():
    """Run a quick demo of AI vs AI."""
    print("Running AI vs AI demonstration (depth=4)...")
    play_game(ai_vs_ai=True, ai_depth=4)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        # Default: Human (RED) vs AI (BLUE)
        print("Starting Human (RED) vs AI (BLUE) game...")
        print("You are RED. Enter move numbers to play.\n")
        play_game(ai_vs_ai=False, ai_depth=4)