class JZO58 {
    class Solution {
        fun reverseLeftWords(s: String, k: Int): String {
            //return s.substring(n) + s.substring(0, n)
            val n = s.length
            val chars = s.toCharArray()
            val k = k % s.length
            var tmp = 'a'
            for (i in 0 until k) {
                tmp = chars[0]
                for (j in 1 until n) {
                    chars[j - 1] = chars[j]
                }
                chars[n - 1] = tmp
            }
            return String(chars)
        }
    }
}