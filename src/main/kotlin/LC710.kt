import java.util.*

fun main() {
    while (true) {
        try {
            LC710.Solution(2, intArrayOf()).pick()
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC710 {
    /// 710. 黑名单中的随机数
    /// 710. 黑名单中的随机数
    class Solution(val n: Int, blacklist: IntArray) {

        val random = Random()

        var bounds: Int
        var blacklistMap = mutableMapOf<Int, Int>()

        init {
            blacklist.sort()
            bounds = n - blacklist.size

            for (b in blacklist) {
                blacklistMap[b] = -1 // 代表存在
            }

            var max = n - 1
            for (b in blacklist) {
                if (b >= bounds) break

                while (blacklistMap.containsKey(max)) max--

                blacklistMap[b] = max--
            }
        }

        fun pick(): Int {
            val randomNext = random.nextInt(bounds)
            if (blacklistMap.containsKey(randomNext)) {
                return blacklistMap[randomNext]!!
            }
            return randomNext
        }
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * var obj = Solution(n, blacklist)
     * var param_1 = obj.pick()
     */
}