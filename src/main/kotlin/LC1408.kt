class LC1408 {
    /// 1408. 数组中的字符串匹配
    class Solution {
        fun stringMatching(words: Array<String>): List<String> {
            val result = mutableListOf<String>()

            for (i in words.indices) {
                val wordI = words[i]
                for (j in words.indices) {
                    if (i == j) continue
                    val wordJ = words[j]
                    if (wordJ.contains(wordI)) {
                        result.add(wordI)
                        break
                    }
                }
            }

            return result
        }
    }
}