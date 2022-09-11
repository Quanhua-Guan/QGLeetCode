import java.util.*

class LC857 {
    /// 857. 雇佣 K 名工人的最低成本
    class Solution {
        fun mincostToHireWorkers(quality: IntArray, wage: IntArray, k: Int): Double {

            // totalWage >= totalQuality * (wage[workers[i]] / quality[workers[i]])

            val n = quality.size
            var workers = Array(n) { it }
            Arrays.sort(workers) { a, b ->
                wage[a] * quality[b] - wage[b] * quality[a]
            }

            val priorityQueue = PriorityQueue<Int> { a, b -> b - a }
            var totalQuality = 0
            for (i in 0 until k - 1) {
                val q = quality[workers[i]]
                totalQuality += q
                priorityQueue.offer(q)
            }
            var minTotalWage = Double.MAX_VALUE
            for (i in k - 1 until n) {
                val q = quality[workers[i]]
                totalQuality += q
                priorityQueue.offer(q)
                minTotalWage = minOf(minTotalWage, totalQuality.toDouble() * (wage[workers[i]].toDouble() / quality[workers[i]].toDouble()))
                totalQuality -= priorityQueue.poll()
            }

            return minTotalWage
        }
    }
}