import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution1000().mergeStones(intArrayOf(3, 5, 1, 2, 2), 3)
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}