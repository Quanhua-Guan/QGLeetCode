class LC256 {
    /// 256. 粉刷房子
    class Solution {
        fun minCost(costs: Array<IntArray>): Int {
            val n = costs.size
            // totalCosts[i][j] 代表将第 i (0 <= i < n) 栋房子粉刷成颜色 j (0 <= j < 3) 的前提下, 前 i+1 栋房子粉刷的最小花费金额.
            val totalCosts = Array(n) { IntArray(3) }
            // 显然:
            for (j in 0 until 3) {
                totalCosts[0][j] = costs[0][j]
            }
            // 可以推导:
            // totalCosts[i][0] = minOf(                      totalCosts[i - 1][1], totalCosts[i - 1][2]) + costs[i][0]
            // totalCosts[i][1] = minOf(totalCosts[i - 1][0],                       totalCosts[i - 1][2]) + costs[i][1]
            // totalCosts[i][2] = minOf(totalCosts[i - 1][0], totalCosts[i - 1][1]                      ) + costs[i][2]
            for (i in 1 until n) {
                totalCosts[i][0] = minOf(totalCosts[i - 1][1], totalCosts[i - 1][2]) + costs[i][0]
                totalCosts[i][1] = minOf(totalCosts[i - 1][0], totalCosts[i - 1][2]) + costs[i][1]
                totalCosts[i][2] = minOf(totalCosts[i - 1][0], totalCosts[i - 1][1]) + costs[i][2]
            }

            // 结果就是 totalCosts[n - 1] 中最小的那个, 把最后移动房子粉刷成 红色, 蓝色, 绿色 的最小话费金额的最小值.
            return totalCosts[n - 1].minOf { it }
        }
    }

    class Solution_空间优化 {
        fun minCost(costs: Array<IntArray>): Int {
            val n = costs.size
            // totalCosts[j]
            val totalCosts = IntArray(3)
            // 显然:
            for (j in 0 until 3) {
                totalCosts[j] = costs[0][j]
            }
            // 可以推导:
            for (i in 1 until n) {
                val tc0 = totalCosts[0]
                val tc1 = totalCosts[1]
                val tc2 = totalCosts[2]

                totalCosts[0] = minOf(tc1, tc2) + costs[i][0]
                totalCosts[1] = minOf(tc0, tc2) + costs[i][1]
                totalCosts[2] = minOf(tc0, tc1) + costs[i][2]
            }

            return minOf(totalCosts[0], totalCosts[1], totalCosts[2])
        }
    }
}