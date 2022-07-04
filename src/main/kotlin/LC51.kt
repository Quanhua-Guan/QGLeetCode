class LC51 {
    /// 51. N 皇后
    class Solution {
        fun solveNQueens(n: Int): List<List<String>> {
            val cols = mutableSetOf<Int>() // 0..n-1      => row
            val diagonal1 = mutableSetOf<Int>() // -n..n  => row - col
            val diagonal2 = mutableSetOf<Int>() // 0..n+n => row + col

            val result = mutableListOf<List<String>>()

            val board = List(n) { CharArray(n) { '.' } }
            fun boardDesc(): List<String> {
                return board.map { it.joinToString("") }
            }

            fun trace(row: Int) {
                if (row == n) {
                    result.add(boardDesc())
                    return
                }

                for (col in 0 until n) {
                    if (!cols.contains(col) && !diagonal1.contains(row - col) && !diagonal2.contains(
                            row + col
                        )
                    ) {
                        board[row][col] = 'Q'
                        cols.add(col)
                        diagonal1.add(row - col)
                        diagonal2.add(row + col)
                        trace(row + 1)
                        board[row][col] = '.'
                        cols.remove(col)
                        diagonal1.remove(row - col)
                        diagonal2.remove(row + col)
                    }
                }
            }

            trace(0)
            return result
        }
    }
}