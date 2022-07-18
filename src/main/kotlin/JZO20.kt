class JZO20 {
    /// 剑指 Offer 20. 表示数值的字符串
    class Solution {
        fun isNumber(s: String): Boolean {
            val states = listOf(
                mapOf<Char, Int>(Pair(' ', 0), Pair('d', 1), Pair('.', 2), Pair('s', 3)), // 0
                mapOf<Char, Int>(Pair('d', 1), Pair('e', 4), Pair('.', 8), Pair(' ', 7)), // 1
                mapOf<Char, Int>(Pair('d', 8)), // 2
                mapOf<Char, Int>(Pair('d', 1), Pair('.', 2)), // 3
                mapOf<Char, Int>(Pair('d', 6), Pair('s', 5)), // 4
                mapOf<Char, Int>(Pair('d', 6)), // 5
                mapOf<Char, Int>(Pair('d', 6), Pair(' ', 7)), // 6
                mapOf<Char, Int>(Pair(' ', 7)), // 7
                mapOf<Char, Int>(Pair('d', 8), Pair('e', 4), Pair(' ', 7)) // 8
            )
            var state = 0
            for (c in s) {
                var action = '?'
                if (c == ' ') action = ' '
                else if (c.isDigit()) action = 'd'
                else if (c == 'e' || c == 'E') action = 'e'
                else if (c == '+' || c == '-') action = 's'
                else if (c == '.') action = '.'

                if (!states[state]!!.containsKey(action)) return false
                state = states[state][action]!!
            }
            return state == 1 || state == 6 || state == 7 || state == 8
        }
    }
}