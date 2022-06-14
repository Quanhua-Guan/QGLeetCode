class LC498 {
    /// 498. 对角线遍历
    class Solution {
        fun findDiagonalOrder(mat: Array<IntArray>): IntArray {
            val rowCount = mat.size
            val colCount = mat[0].size

            var r = 0
            var c = 0
            var dr = -1
            var dc = 1

            val result = IntArray(rowCount * colCount)
            var i = 0

            while (r in 0 until rowCount && c in 0 until colCount) {
                result[i++] = mat[r][c]
                r += dr
                c += dc
                if (r < 0 && c in 0 until colCount) {
                    r = 0
                    dr = 1
                    dc = -1
                } else if (r >= rowCount && c in 0 until colCount) {
                    r = rowCount - 1
                    c += 2
                    dr = -1
                    dc = 1
                } else if (r < 0 && c >= colCount) {
                    r += 2
                    c -= 1
                    dr = 1
                    dc = -1
                } else if (r >= rowCount && c < 0) {
                    r -= 1
                    c += 2
                    dr = -1
                    dc = 1
                } else if (c < 0 && r in 0 until rowCount) {
                    c = 0
                    dr = -1
                    dc = 1
                } else if (c >= colCount && r in 0 until rowCount) {
                    r += 2
                    c -= 1
                    dr = 1
                    dc = -1
                }
            }
            return result
        }
    }
}