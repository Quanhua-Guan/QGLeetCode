import java.util.*

fun main() {
    while (true) {
        try {
            println(
                LC636.Solution().exclusiveTime(
                    1,
                    listOf("0:start:0", "0:start:2", "0:end:5", "0:start:6", "0:end:6", "0:end:7")
                )
            )
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC636 {
    class Solution {
        fun exclusiveTime(n: Int, logs: List<String>): IntArray {
            val intervals = IntArray(n)
            val logStack = LinkedList<Triple<Int, Boolean, Int>>();

            fun getLogInfo(log: String): Triple<Int, Boolean, Int> {
                var logInfo = log.split(":")
                return Triple(logInfo[0].toInt(), logInfo[1] == "start", logInfo[2].toInt());
            }

            for (log in logs) {
                val logInfo = getLogInfo(log)
                val (funId, isStart, time) = logInfo

                if (logStack.isNotEmpty()) {
                    val (prevFunId, prevIsStart, prevTime) = logStack.peek()
                    intervals[prevFunId] += time - prevTime + (if (prevFunId == funId && prevIsStart && !isStart) 1 else 0)
                }

                if (isStart) {
                    logStack.push(logInfo)
                } else {
                    logStack.pop()
                    if (logStack.isNotEmpty()) {
                        val (prevFunId, prevIsStart, prevTime) = logStack.pop()
                        logStack.push(Triple(prevFunId, prevIsStart, time + 1))
                    }
                }
            }

            return intervals
        }
    }
}