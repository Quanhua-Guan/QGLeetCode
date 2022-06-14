class LC149 {
    /// 149. 直线上最多的点数
    class Solution {
        fun gcd(a: Int, b: Int): Int {
            return if (b == 0) a else gcd(b, a % b)
        }
        fun maxPoints(points: Array<IntArray>): Int {
            val n = points.size
            if (n <= 2) return n

            var maxCount = 0
            for (i in 0 until n) { // 第 i 次迭代可以找到的最多共线点数为 n - i 个, 依次 n, n-1, n-2...1
                if (maxCount > n / 2 || maxCount >= n - i) break // 剪枝, 不可能找到更大的 maxCount 时, 提前退出

                val countMap = mutableMapOf<Int, Int>() // 记录以点 points[i] 为基准点, 可以构成的不同斜率的数量
                for (j in i + 1 until n) {
                    var x = points[i][0] - points[j][0]
                    var y = points[i][1] - points[j][1]
                    if (x == 0) { // 为了最终计算的 key 值一致
                        y = 1
                    } else if (y == 0) { // 为了最终计算的 key 值一致
                        x = 1
                    } else {
                        if (x < 0) { // 为了最终计算的 key 值一致, 保证斜率相同的两个点会被统计到同一个 key 值上.
                            x = -x
                            y = -y
                        }
                        val g = gcd(x, y) // 求最大公约数
                        x /= g
                        y /= g
                    }
                    val key = y + x * 20001
                    countMap[key] = countMap.getOrDefault(key, 0) + 1
                }
                var tmpMaxCount = 0
                for ((_, v) in countMap) {
                    tmpMaxCount = maxOf(tmpMaxCount, v + 1) // 这里的 + 1 代表的是当前迭代的 points[i] 点
                }

                maxCount = maxOf(maxCount, tmpMaxCount)
            }

            return maxCount
        }
    }
}