fun main() {
    while (true) {
        try {
            println(
                LC30.Solution().findSubstring("wordgoodgoodgoodbestword", arrayOf("word","good","best","word"))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC30 {
    class Solution {
        fun findSubstring(s: String, words: Array<String>): List<Int> {
            // 整理word出现次数的哈希表
            val wordsMap = mutableMapOf<String, Int>()
            val remain = mutableMapOf<String, Int>()
            for (word in words) {
                val count = wordsMap.getOrDefault(word, 0) + 1
                wordsMap[word] = count
                remain[word] = count
            }

            fun resetRemainMap() {
                for ((k, v) in wordsMap) {
                    remain[k] = v
                }
            }

            // 遍历s
            val sl = s.length
            val wl = words[0].length
            val wn = words.size

            val results = mutableListOf<Int>()


            var start = 0
            var current = 0
            while (current + wl <= sl) {
                val ss = s.substring(current, current + wl)
                if (!remain.containsKey(ss)) {
                    start += 1
                    if (start + wl * wn > sl) break
                    current = start
                    resetRemainMap()
                } else {
                    val newCount = remain[ss]!! - 1
                    if (newCount == 0) {
                        remain.remove(ss)
                    } else {
                        remain[ss] = newCount
                    }

                    if (current - start + wl == wl * wn) {
                        // 匹配成功
                        results.add(start)

                        start += 1
                        if (start + wl * wn > sl) break
                        current = start
                        resetRemainMap()
                    } else {
                        current += wl
                        if (current >= sl) {
                            start = start + 1
                            if (start + wl * wn > sl) break
                            current = start
                            resetRemainMap()
                        }
                    }
                }
            }
            return results
        }
    }
}