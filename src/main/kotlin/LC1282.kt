class LC1282 {
    /// 1282. 用户分组
    class Solution {
        fun groupThePeople(groupSizes: IntArray): List<List<Int>> {
            val result = mutableListOf<List<Int>>()
            val resultMap = mutableMapOf<Int, MutableList<Int>>()

            for (i in groupSizes.indices) {
                val groupSize = groupSizes[i]
                var list: MutableList<Int>
                if (resultMap.containsKey(groupSize)) {
                    list = resultMap[groupSize]!!
                } else {
                    list = mutableListOf<Int>()
                    resultMap[groupSize] = list
                }

                list.add(i)
                if (list.size == groupSize) {
                    resultMap.remove(groupSize)
                    result.add(list)
                }
            }

            return result
        }
    }
}