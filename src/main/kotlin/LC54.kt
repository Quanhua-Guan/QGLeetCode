class LC54 {
    class Solution {
        fun spiralOrder(matrix: Array<IntArray>): List<Int> {
            val rowCount = matrix.size
            if (rowCount == 0) return listOf()

            val colCount = matrix[0].size
            val count = rowCount * colCount
            val nums = mutableListOf<Int>()

            var r = 0
            var c = 0

            val visit = Array(rowCount) { BooleanArray(colCount) }
            var direction = 0 // 0 right, 1 down, 2 left, 3 up
            while (nums.size < count) {
                nums.add(matrix[r][c])
                visit[r][c] = true
                val rr = r
                val cc = c

                when (direction) {
                    0 -> c++
                    1 -> r++
                    2 -> c--
                    3 -> r--
                }

                if (!(r in 0 until rowCount && c in 0 until colCount) || visit[r][c]) {
                    direction = (direction + 1) % 4
                    r = rr
                    c = cc

                    when (direction) {
                        0 -> c++
                        1 -> r++
                        2 -> c--
                        3 -> r--
                    }
                }
            }

            return nums
        }

        fun spiralOrder_1(matrix: Array<IntArray>): List<Int> {
            val rowCount = matrix.size
            val colCount = matrix[0].size
            val nums = IntArray(rowCount * colCount)

            var left = 0
            var right = colCount - 1
            var top = 0
            var bottom = rowCount - 1
            var i = 0

            while (left <= right && top <= bottom) {
                // left -> right
                for (c in left..right) {
                    nums[i++] = matrix[top][c]
                }
                // top -> bottom
                for (r in top + 1 until bottom) {
                    nums[i++] = matrix[r][right]
                }
                // right -> left
                if (top != bottom) {
                    for (c in right downTo left) {
                        nums[i++] = matrix[bottom][c]
                    }
                }
                // bottom -> top
                if (left != right) {
                    for (r in bottom - 1 downTo top + 1) {
                        nums[i++] = matrix[r][left]
                    }
                }

                left++
                right--
                top++
                bottom--
            }

            return nums.toList()
        }
    }
}