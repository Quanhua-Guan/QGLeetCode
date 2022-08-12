import java.lang.Math.abs
import java.lang.StringBuilder

class LC1417 {
    /// 1417. 重新格式化字符串
    class Solution {
        fun reformat(s: String): String {
            val charList = mutableListOf<Char>()
            var digitList = mutableListOf<Char>()
            for (c in s) {
                if (c.isDigit()) {
                    digitList.add(c)
                } else if (c.isLowerCase()) {
                    charList.add(c)
                } else {
                    assert(false) { "invalid input string" }
                }
            }

            if (Math.abs(charList.size - digitList.size) > 1) return ""

            var listLong: MutableList<Char>
            var listShort: MutableList<Char>
            if (charList.size > digitList.size) {
                listLong = charList
                listShort = digitList
            } else {
                listLong = digitList
                listShort = charList
            }

            val result = StringBuilder()
            var iL = 0
            var iS = 0
            var useLong = true
            while (iL < listLong.size || iS < listShort.size) {
                if (useLong) {
                    result.append(listLong[iL++])
                } else {
                    result.append(listShort[iS++])
                }
                useLong = !useLong
            }

            return result.toString()
        }
    }
}