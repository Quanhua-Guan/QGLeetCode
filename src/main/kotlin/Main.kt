import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution675().cutOffTree(
                    listOf(listOf(1,2,3),listOf(0,0,0),listOf(7,6,5))
                )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}