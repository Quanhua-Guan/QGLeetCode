import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(
                Solution525().findMaxLength_preSum(intArrayOf(0,1,0,1,1,0,1,0))
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}