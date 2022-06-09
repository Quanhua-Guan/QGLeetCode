import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution413().numberOfArithmeticSlices(intArrayOf(1, 1, 1, 1, 2, 3, 3, 3, 3))
            )
            val sol = Solution497(arrayOf(intArrayOf(82918473,-57180867,82918476,-57180863),intArrayOf(83793579,18088559,83793580,18088560),intArrayOf(66574245,26243152,66574246,26243153),intArrayOf(72983930,11921716,72983934,11921720)))
            var i = 0
            while (i++ < 10000) {
                val point = sol.pick()
                if (!sol.check(point)) {
                    println(point)
                }
            }
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}