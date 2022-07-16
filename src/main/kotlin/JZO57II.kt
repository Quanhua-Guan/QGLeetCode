class JZO57II {
    /// 剑指 Offer 57 - II. 和为s的连续正数序列
    class Solution {
        fun findContinuousSequence(target: Int): Array<IntArray> {
            val results = mutableListOf<IntArray>()
            val n = target

            var left = 1
            var right = 2
            var sum = left
            var doAdding = true
            while (left < right && right < target) {
                if (doAdding) {
                    sum += right
                }
                if (sum == target) {
                    results.add((left..right).map { it }.toIntArray())

                    sum -= left
                    left++
                    right++
                    doAdding = true
                } else if (sum > target) {
                    sum -= left
                    left++
                    doAdding = false
                } else { // sum < target
                    right++
                    doAdding = true
                }
            }

            return Array<IntArray>(results.size) { results[it] }
        }

        fun findContinuousSequence1(target: Int): Array<IntArray> {
            val results = mutableListOf<IntArray>()
            val n = target

            var left = 1
            var right = 2
            while (left < right && right < target) {
                val sum = (left + right) * (right - left + 1) / 2
                if (sum == target) {
                    results.add((left..right).map { it }.toIntArray())
                    left++
                    right++
                } else if (sum > target) {
                    left++
                } else { // sum < target
                    right++
                }
            }

            return Array<IntArray>(results.size) { results[it] }
        }
    }
}