class JZO58I {
    /// 剑指 Offer 58 - I. 翻转单词顺序
    class Solution {
        fun reverseWords(s: String): String {
            val words = mutableListOf<String>()
            val charList = mutableListOf<Char>()

            for (c in s) {
                if (c != ' ') {
                    charList.add(c)
                } else {
                    if (charList.isNotEmpty()) {
                        words.add(charList.joinToString(""))
                        charList.clear()
                    }
                }
            }

            if (charList.isNotEmpty()) {
                words.add(charList.joinToString(""))
                charList.clear()
            }

            words.reverse()

            return words.joinToString(" ")
        }
    }
}