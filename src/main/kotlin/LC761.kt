
fun main() {
    while (true) {
        try {
            println(
                LC761.Solution().makeLargestSpecial("11011000")
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}


class LC761 {
    /// 761. 特殊的二进制序列
    class Solution {
        val mem = mutableMapOf<String, String>()
        fun makeLargestSpecial(s: String): String {
            if (s.isEmpty()) return s

            if (mem.containsKey(s)) {
                return mem[s]!!
            }

            val substrs = mutableListOf<String>()
            var x = 0
            var y = 0
            var start = x
            for (c in s) {
                ++x
                y += (if (c == '1') 1 else -1)

                if (y == start) {
                    substrs.add(s.substring(y, x))
                    // 更新 y, start 为 x
                    // 继续寻找下一个特殊子串
                    y = x
                    start = x
                }
            }

            for (i in substrs.indices) {
                val substr = substrs[i]

                // 拆出第一个字符, 它必然是 "1"
                assert(substr.substring(0, 1) == "1")
                // 拆出最后一个字符, 它必然是 "0"
                assert(substr.substring(substr.length - 1, substr.length) == "0")

                substrs[i] = "1" + makeLargestSpecial(substr.substring(1, substr.length - 1)) + "0"
            }

            substrs.sortDescending()
            return substrs.joinToString("").also { mem[s] = it }
        }
    }
}