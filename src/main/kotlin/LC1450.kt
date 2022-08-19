class LC1450 {
    /// 1450. 在既定时间做作业的学生人数
    class Solution {
        fun busyStudent(startTime: IntArray, endTime: IntArray, queryTime: Int): Int {
            var count = 0
            for (i in startTime.indices) {
                val start = startTime[i]
                val end = endTime[i]
                if (start <= queryTime && queryTime <= end) {
                    count++
                }
            }
            return count
        }
    }
}