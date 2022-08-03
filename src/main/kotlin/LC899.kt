import java.util.*

class LC899 {
    /// 899. 有序队列
    class Solution {
        fun orderlyQueue(s: String, k: Int): String {
            var charList = s.map { it }

            // k >= 2 时, 代表字符串中任意两个字符可以对调顺序, 最小结果就是升序排序结果.
            if (k >= 2) return charList.sorted().joinToString("")

            // k == 1 时, 最小结果肯定是以字符串中最小字符大头的字符串, 依次遍历去最小即可.
            var minChar = Collections.min(charList)
            var minString = s
            for (i in s.indices) {
                val c = s[i]
                if (c == minChar) {
                    // 由于每次只能将第一个字符移动到最后, 如果希望将 i 个字符最终在下标 0 位置,
                    // 则可以将下标 [0..(i-1)] 的字符依次移动到最后.
                    // 这个过程, 相当于子串 s[i..(n-1)] 连接 子串 s[0..(i - 1)].
                    // (说明: n代表字符串长度; 另外注意这里用的是闭区间, 代表包含区间两端下标对应的字符)
                    val cString = s.substring(i) + s.substring(0, i)
                    if (cString.compareTo(minString) < 0) {
                        minString = cString
                    }
                }
            }

            return minString
        }
    }
}