import kotlin.math.min

fun main() {
    while (true) {
        try {
            println(
                LC741.Solution().cherryPickup(arrayOf(
                    intArrayOf(1,1,1,1,0,0,0),
                    intArrayOf(0,0,0,1,0,0,0),
                    intArrayOf(0,0,0,1,0,0,1),
                    intArrayOf(1,0,0,1,0,0,0),
                    intArrayOf(0,0,0,1,0,0,0),
                    intArrayOf(0,0,0,1,0,0,0),
                    intArrayOf(0,0,0,1,1,1,1),
                ))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }
}

class LC741 {
    /// 741. 摘樱桃
    class Solution {
        fun cherryPickup(grid: Array<IntArray>): Int {
            val n = grid.size

            // 定义 f[k][x1][x2] 表示 2 个人从 (0, 0) 出发, 走过 k 步, 分别走到 (x1, k - x1) 和 (x2, k - x2) 时,
            // 摘到的樱桃数之和的最大值.
            val f = Array(n * 2 - 1) { Array(n) { IntArray(n) { Int.MIN_VALUE } } }
            // 显然
            f[0][0][0] = grid[0][0]

            for (k in 1 until n * 2 - 1) {
                for (x1 in maxOf(k - (n - 1), 0)..minOf(k, n - 1)) {
                    val y1 = k - x1
                    if (grid[x1][y1] == -1) {
                        continue
                    }

                    for (x2 in x1..minOf(k, n - 1)) {
                        val y2 = k - x2
                        if (grid[x2][y2] == -1) {
                            continue
                        }

                        var res = f[k - 1][x1][x2] // 都往右
                        if (x1 > 0) {
                            res = maxOf(res, f[k - 1][x1 - 1][x2])
                        }
                        if (x2 > 0) {
                            res = maxOf(res, f[k - 1][x1][x2 - 1])
                        }
                        if (x1 > 0 && x2 > 0) {
                            res = maxOf(res, f[k - 1][x1 - 1][x2 - 1])
                        }
                        res += grid[x1][y1]
                        if (x1 != x2) {
                            res += grid[x2][y2]
                        }
                        f[k][x1][x2] = res
                    }
                }
            }
            return maxOf(0, f[2 * n - 2][n - 1][n - 1])
        }
    }
}