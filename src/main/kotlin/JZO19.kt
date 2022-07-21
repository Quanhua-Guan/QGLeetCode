class JZO19 {
    /// 剑指 Offer 19. 正则表达式匹配
    class Solution {
        fun isMatch(s: String, p: String): Boolean {
            // 解析匹配字符串数组
            val patterns = mutableListOf<CharArray>()
            var i = 0
            var prevChar: Char? = null
            while (i < p.length) {
                if (prevChar == null) {
                    prevChar = p[i]
                } else {
                    if (p[i] == '*') {
                        patterns.add(charArrayOf(prevChar, '*'))
                        prevChar = null
                    } else {
                        patterns.add(charArrayOf(prevChar))
                        prevChar = p[i]
                    }
                }
                i++
            }
            if (prevChar != null) {
                patterns.add(charArrayOf(prevChar))
            }

            // match[i][j] 代表 s[0..<i] 与 patterns[0..<j] 是否匹配
            val match = Array(s.length + 1) { BooleanArray(patterns.size + 1) }
            match[0][0] = true // 空匹配空
            for (j in 1 until patterns.size + 1) {
                match[0][j] = match[0][j - 1] && isMatchEmpty(patterns[j - 1])
            }

            for (i in 1 until s.length + 1) {
                for (j in 1 until patterns.size + 1) {
                    match[i][j] =
                        (match[i - 1][j] && isMatchEmpty(patterns[j - 1]) && isCharMatchPattern(
                            s[i - 1],
                            patterns[j - 1]
                        )) ||
                                (match[i][j - 1] && isMatchEmpty(patterns[j - 1])) ||
                                (match[i - 1][j - 1] && isCharMatchPattern(
                                    s[i - 1],
                                    patterns[j - 1]
                                ))
                }
            }

            return match[s.length][patterns.size]
        }

        fun isMatchEmpty(pattern: CharArray): Boolean {
            return pattern.size == 2 && pattern[1] == '*'
        }

        fun isCharMatchPattern(c: Char, pattern: CharArray): Boolean {
            if (pattern.isEmpty()) return false
            if (pattern.size == 1) {
                return pattern[0] == '.' || pattern[0] == c
            }
            assert(pattern.size == 2)
            return (pattern[0] == c || pattern[0] == '.') && pattern[1] == '*'
        }
    }
}