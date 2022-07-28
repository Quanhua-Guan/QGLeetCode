class LC1331 {
    /// 1331. 数组序号转换
    class Solution {
        fun arrayRankTransform(arr: IntArray): IntArray {
            var index = 0
            var prevNum = Int.MIN_VALUE
            arr.mapIndexed { index, it ->  Pair(it, index)}.sortedBy { it.first }.forEach { (num, prevIndex) ->
                if (prevNum != num) ++index
                arr[prevIndex] = index
                prevNum = num
            }
            return arr
        }
    }
}