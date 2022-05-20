import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            val mat = arrayOf(
                intArrayOf(2, 1, 1),
                intArrayOf(0, 1, 1),
                intArrayOf(1, 0, 1)
            )
            println(Solution994().orangesRotting(mat))
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}