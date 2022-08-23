fun main() {
    while (true) {
        try {
            println(
                LC782.Solution().movesToChessboard(
                    arrayOf(
                        intArrayOf(1, 0, 0, 1),
                        intArrayOf(1, 0, 0, 1),
                        intArrayOf(0, 1, 1, 0),
                        intArrayOf(0, 1, 1, 0)
                    )
                )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC782 {
    /// 782. 变为棋盘
    class Solution {
        fun movesToChessboard(board: Array<IntArray>): Int {
            val n = board.size

            fun getMask(index: Int): Pair<Int, Int> {
                var rowMask = 0
                var colMask = 0
                for (i in 0 until n) {
                    // 第一行
                    rowMask = rowMask or (board[index][i] shl i)
                    // 第一列
                    colMask = colMask or (board[i][index] shl i)
                }
                return Pair(rowMask, colMask)
            }

            // 获取第一行 rowMask, 第一列 colMask
            val (rowMask, colMask) = getMask(0)
            val reverseRowMask = ((1 shl n) - 1) xor rowMask
            val reverseColMask = ((1 shl n) - 1) xor colMask

            var rowMaskCount = 0 // 和第一行相同的行数
            var colMaskCount = 0 // 和第一列相同的列数
            for (i in 0 until n) {
                val (curRowMask, curColMask) = getMask(i)

                if (curRowMask != rowMask && curRowMask != reverseRowMask) {
                    return -1
                } else if (curRowMask == rowMask) {
                    ++rowMaskCount
                }

                if (curColMask != colMask && curColMask != reverseColMask) {
                    return -1
                } else if (curColMask == colMask) {
                    ++colMaskCount
                }
            }

            val rowSwitchCount = getSwitchCount(rowMask, rowMaskCount, n)
            if (rowSwitchCount == -1) return -1
            val colSwitchCount = getSwitchCount(colMask, colMaskCount, n)
            if (colSwitchCount == -1) return -1
            return rowSwitchCount + colSwitchCount
        }

        /// 获取行/列最小交换次数
        fun getSwitchCount(mask: Int, maskCount: Int, n: Int): Int {
            val bit1Count = bitCount(mask)
            if (n and 1 == 0) { // 偶数
                if (bit1Count != n / 2 || maskCount != n / 2) {
                    return -1
                }
                // 开头选择放0, 则需要找出所有偶数位下标上1的个数, 把这些1替换成0即可
                val count0AtFirst = n / 2 - bitCount(mask and 0xAA_AA_AA_AA.toInt())
                val count1AtFirst = n / 2 - bitCount(mask and 0x55_55_55_55)
                return minOf(count0AtFirst, count1AtFirst)
            } else { // 奇数
                if (Math.abs(n - 2 * bit1Count) != 1 || Math.abs(n - 2 * maskCount) != 1) {
                    return -1
                }
                if (bit1Count == n / 2) { // bit 1 个数比 bit 0 个数少1 => 0 开头, 010...010
                    // 偶数位总数 - 偶数位上1的个数 => 偶数位上0的个数 -> 只需把这些0转变为1即可
                    return n / 2 - bitCount(mask and 0xAA_AA_AA_AA.toInt())
                } else { // bit 0 个数比 bit 1 个数少1 => 1 开头, 101...101
                    // 奇数位总数 - 奇数位上1的个数 => 奇数位上0的个数 -> 只需把这些0转变为1即可
                    return (n + 1) / 2 - bitCount(mask and 0x55_55_55_55)
                }
            }
        }

        fun bitCount(i: Int): Int {
            // HD, Figure 5-2
            var i = i
            i -= (i ushr 1 and 0x55555555)
            i = (i and 0x33333333) + (i ushr 2 and 0x33333333)
            i = i + (i ushr 4) and 0x0f0f0f0f
            i += (i ushr 8)
            i += (i ushr 16)
            return i and 0x3f
        }
    }
}