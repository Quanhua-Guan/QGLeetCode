import com.cqmh.qgleetcode.*

fun main() {
    while (true) {
        try {
            println(Solution10_20220521().isMatch("mississippi", "mis*is*ip*."))
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}