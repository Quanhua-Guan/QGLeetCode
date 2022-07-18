import java.util.*

class JZO31 {
    class Solution {
        fun validateStackSequences(pushed: IntArray, popped: IntArray): Boolean {
            var pushIndex = 0
            var popIndex = 0
            val stack = LinkedList<Int>()

            while (pushIndex < pushed.size && popIndex < popped.size) {
                if (stack.isNotEmpty() && stack.peek() == popped[popIndex]) {
                    popIndex++
                } else {
                    stack.push(pushed[pushIndex])
                    pushIndex++
                }
            }

            return stack.isEmpty()
        }
    }
}