import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution413().numberOfArithmeticSlices(intArrayOf(1, 1, 1, 1, 2, 3, 3, 3, 3))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}