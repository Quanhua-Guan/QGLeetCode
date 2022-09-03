import java.util.*

class LC646 {
    /// 646. 最长数对链
    class Solution {
        fun findLongestChain(pairs: Array<IntArray>): Int {
            // 排序
            Arrays.sort(pairs, kotlin.Comparator<IntArray> { p1, p2 ->
                if (p1[0] != p2[0]) p1[0] - p2[0] else p1[1] - p2[1]
            })

            // 定义 min[len] 为长度为 len 的数对链最小的末尾数对 (数对大小, 通过数对的后位数来对比)
            val min = mutableListOf<IntArray>()
            var len = 1
            min.add(intArrayOf(Int.MIN_VALUE, Int.MIN_VALUE))
            min.add(pairs[0])

            for(i in 1..pairs.lastIndex) {
                val pair = pairs[i]
                if (pair[0] > min.last()!![1]) {
                    ++len
                    min.add(pair)
                } else { // pair[0] <= min.last()!![1]
                    val target = pair[1]
                    if (target > min.last()!![1]) {
                        continue
                    }

                    var l = 1
                    var r = len
                    while (l <= r) {
                        var m = (l + r) ushr 1
                        if (min[m][1] < target) {
                            l = m + 1
                        } else {
                            r = m - 1
                        }
                    }
                    min[r + 1] = pair
                }
            }

            return len
        }
    }
}