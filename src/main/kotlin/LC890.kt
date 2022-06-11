fun main() {
    while (true) {
        try {
            // ["abc","deq","mee","aqq","dkd","ccc"]
            // "abb"
            println(
                LC890.Solution().findAndReplacePattern(arrayOf("abc","deq","mee","aqq","dkd","ccc"), "abb")
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC890 {
    class Solution {
        fun findAndReplacePattern(words: Array<String>, pattern: String): List<String> {
            val results = mutableListOf<String>()
            for (word in words) {
                if (word.length != pattern.length) continue

                val patternWordMap = mutableMapOf<Char, Char>()
                val wordPatternMap = mutableMapOf<Char, Char>()
                var valid = true
                for (i in word.indices) {
                    val cp = pattern[i]
                    val cw = word[i]

                    if (patternWordMap.containsKey(cp)) {
                        if (patternWordMap[cp]!! != cw) {
                            valid = false
                            break
                        }
                    } else {
                        patternWordMap[cp] = cw
                    }

                    if (wordPatternMap.containsKey(cw)) {
                        if (wordPatternMap[cw]!! != cp) {
                            valid = false
                            break
                        }
                    } else {
                        wordPatternMap[cw] = cp
                    }
                }
                if (valid) {
                    results.add(word)
                }
            }
            return results
        }
    }
}