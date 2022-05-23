import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution128().longestConsecutive(
                    intArrayOf(
                        -4,
                        -1,
                        4,
                        -5,
                        1,
                        -6,
                        9,
                        -6,
                        0,
                        2,
                        2,
                        7,
                        0,
                        9,
                        -3,
                        8,
                        9,
                        -2,
                        -6,
                        5,
                        0,
                        3,
                        4,
                        -2
                    )
                )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}