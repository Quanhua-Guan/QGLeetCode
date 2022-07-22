class LC757 {
    /// 757. 设置交集大小至少为2
    class Solution {
        fun intersectionSizeTwo(intervals: Array<IntArray>): Int {
            intervals.sortWith( Comparator { a, b -> if (a[1] == b[1]) b[0] - a[0] else a[1] - b[1] } )
            var count = 0
            var secondMax = -1
            var max = -1
            for (i in intervals) {
                if (max < i[0]) {
                    secondMax = i[1] - 1
                    max = i[1]
                    count += 2
                } else if (secondMax < i[0]) { // secondMax < i[0] <= max
                    secondMax = max
                    max = i[1]
                    count += 1
                }
            }
            return count
        }
    }
}