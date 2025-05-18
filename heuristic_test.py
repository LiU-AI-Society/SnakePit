import pytest
import numpy as np
import torch
from heuristic import heuristic

def create_empty_board(height=11, width=11):
    """Create an empty board with given dimensions"""
    return np.zeros((height, width, 3), dtype=np.int8)

def test_empty_board():
    """Test when no snake is present"""
    board = create_empty_board()
    valid_moves, moves, action_mask = heuristic(board)
    
    assert len(valid_moves) == 0
    assert all(moves.values())  # All moves should be True
    assert torch.all(action_mask == torch.tensor([True, True, True, True]))

def test_wall_collisions():
    """Test wall collision detection"""
    # Test top wall
    board = create_empty_board(5, 5)
    board[0, 2, 1] = 5  # Place head at top edge
    valid_moves, moves, _ = heuristic(board)
    assert "up" not in valid_moves
    assert not moves["up"]
    
    # Test bottom wall
    board = create_empty_board(5, 5)
    board[4, 2, 1] = 5  # Place head at bottom edge
    valid_moves, moves, _ = heuristic(board)
    assert "down" not in valid_moves
    assert not moves["down"]
    
    # Test left wall
    board = create_empty_board(5, 5)
    board[2, 0, 1] = 5  # Place head at left edge
    valid_moves, moves, _ = heuristic(board)
    assert "left" not in valid_moves
    assert not moves["left"]
    
    # Test right wall
    board = create_empty_board(5, 5)
    board[2, 4, 1] = 5  # Place head at right edge
    valid_moves, moves, _ = heuristic(board)
    assert "right" not in valid_moves
    assert not moves["right"]

def test_neck_detection():
    """Test neck collision detection"""
    board = create_empty_board()
    # Place snake head and one body segment
    board[5, 5, 1] = 5  # Head
    board[5, 4, 1] = 1  # Body segment to left
    
    valid_moves, moves, _ = heuristic(board)
    assert "left" not in valid_moves
    assert not moves["left"]

def test_action_mask_format():
    """Test action mask format and ordering"""
    board = create_empty_board()
    board[5, 5, 1] = 5  # Head in middle
    
    _, _, action_mask = heuristic(board)
    
    assert isinstance(action_mask, torch.Tensor)
    assert action_mask.dtype == torch.bool
    assert action_mask.shape == (4,)



if __name__ == "__main__":
    pytest.main([__file__])