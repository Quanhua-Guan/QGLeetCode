class LC1455 {
    /// 1455. 检查单词是否为句中其他单词的前缀
    class Solution {
        fun isPrefixOfWord(sentence: String, searchWord: String): Int {
            fun isPrefix(start: Int, end: Int): Boolean {
                if (end - start < searchWord.length) return false
                for (i in searchWord.indices) {
                    if (sentence[i + start] != searchWord[i]) return false
                }
                return true
            }

            var start = 0
            var index = 1
            for (i in sentence.indices) {
                val c = sentence[i]
                if (c == ' ' || i == sentence.length - 1) {
                    if (isPrefix(start, if (i == sentence.length - 1) i + 1 else i)) {
                        return index
                    }
                    start = i + 1
                    index++
                }
            }

            return -1
        }
    }
}