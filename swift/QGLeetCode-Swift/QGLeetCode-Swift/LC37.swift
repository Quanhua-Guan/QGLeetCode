//
//  LC37.swift
//  QGLeetCode-Swift
//
//  Created by Quanhua on 2023/1/28.
//

import Foundation

/*
 class Solution {
     private boolean[][] line = new boolean[9][9];
     private boolean[][] column = new boolean[9][9];
     private boolean[][][] block = new boolean[3][3][9];
     private boolean valid = false;
     private List<int[]> spaces = new ArrayList<int[]>();

     public void solveSudoku(char[][] board) {
         for (int i = 0; i < 9; ++i) {
             for (int j = 0; j < 9; ++j) {
                 if (board[i][j] == '.') {
                     spaces.add(new int[]{i, j});
                 } else {
                     int digit = board[i][j] - '0' - 1;
                     line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = true;
                 }
             }
         }

         dfs(board, 0);
     }

     public void dfs(char[][] board, int pos) {
         if (pos == spaces.size()) {
             valid = true;
             return;
         }

         int[] space = spaces.get(pos);
         int i = space[0], j = space[1];
         for (int digit = 0; digit < 9 && !valid; ++digit) {
             if (!line[i][digit] && !column[j][digit] && !block[i / 3][j / 3][digit]) {
                 line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = true;
                 board[i][j] = (char) (digit + '0' + 1);
                 dfs(board, pos + 1);
                 line[i][digit] = column[j][digit] = block[i / 3][j / 3][digit] = false;
             }
         }
     }
 }

 作者：力扣官方题解
 链接：https://leetcode.cn/problems/sudoku-solver/solutions/414120/jie-shu-du-by-leetcode-solution/
 来源：力扣（LeetCode）
 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
 */

class LC37 {
    class Solution {
        private var rows = [[Bool]].init(repeating: [Bool].init(repeating: false, count: 9), count: 9)
        private var cols = [[Bool]].init(repeating: [Bool].init(repeating: false, count: 9), count: 9)
        private var boxes = [[[Bool]]].init(repeating: [[Bool]].init(repeating: [Bool].init(repeating: false, count: 9), count: 3), count: 3)
        
        private var valid = false
        private var spaces = [(Int, Int)]()
        let charDot = Character(".")
        let charZero = Character("0")
        
        func solveSudoku(_ board: inout [[Character]]) {
            for i in 0..<9 {
                for j in 0..<9 {
                    if board[i][j] == charDot {
                        spaces.append((i, j))
                    } else {
                        let digit = Int(board[i][j].asciiValue! - charZero.asciiValue! - 1)
                        rows[i][digit] = true
                        cols[j][digit] = true
                        boxes[i / 3][j / 3][digit] = true
                    }
                }
            }
            
            dfs(&board, 0)
        }
        
        func dfs(_ board: inout [[Character]], _ pos: Int) {
            if pos == spaces.count {
                valid = true
                return
            }
            
            let (i, j) = spaces[pos]
            for digit in 0..<9 {
                if valid { break }
                
                if !rows[i][digit] && !cols[j][digit] && !boxes[i / 3][j / 3][digit] {
                    rows[i][digit] = true
                    cols[j][digit] = true
                    boxes[i / 3][j / 3][digit] = true
                    
                    board[i][j] = Character(UnicodeScalar(UInt8(digit + Int(charZero.asciiValue!) + 1)))
                    dfs(&board, pos + 1)
                    
                    rows[i][digit] = false
                    cols[j][digit] = false
                    boxes[i / 3][j / 3][digit] = false
                }
            }
        }
    }
}
