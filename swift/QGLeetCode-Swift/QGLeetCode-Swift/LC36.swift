//
//  LC36.swift
//  QGLeetCode-Swift
//
//  Created by Quanhua on 2023/1/28.
//

import Foundation

/*
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int[][] rows = new int[9][9];
        int[][] columns = new int[9][9];
        int[][][] subboxes = new int[3][3][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c != '.') {
                    int index = c - '0' - 1;
                    rows[i][index]++;
                    columns[j][index]++;
                    subboxes[i / 3][j / 3][index]++;
                    if (rows[i][index] > 1 || columns[j][index] > 1 || subboxes[i / 3][j / 3][index] > 1) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
*/

class LC36 {
    
    class Solution {
        func isValidSudoku(_ board: [[Character]]) -> Bool {
            var rows = [[Int]].init(repeating: [Int].init(repeating: 0, count: 9), count: 9)
            var cols = [[Int]].init(repeating: [Int].init(repeating: 0, count: 9), count: 9)
            var boxes = [[[Int]]].init(repeating: [[Int]].init(repeating: [Int].init(repeating: 0, count: 9), count: 3), count: 3)
            
            let zeroCharValue = Character("0").asciiValue!
            for i in 0..<9 {
                for j in 0..<9 {
                    let c = board[i][j]
                    if c == Character(".") {
                        continue
                    }
                    
                    let index = Int(c.asciiValue! - zeroCharValue - 1)
                    rows[i][index] += 1
                    cols[j][index] += 1
                    boxes[i / 3][j / 3][index] += 1
                    
                    if rows[i][index] > 1 || cols[j][index] > 1 || boxes[i / 3][j / 3][index] > 1{
                        return false
                    }
                }
            }
            
            return true
        }
    }
    
}
