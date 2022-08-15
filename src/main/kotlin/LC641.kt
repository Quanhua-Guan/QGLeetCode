fun main() {
    while (true) {
        /*
        ["MyCircularDeque","insertLast","insertLast","insertFront","insertFront","getRear","isFull","deleteLast","insertFront","getFront","getRear"]
[[3],[1],[2],[3],[4],[],[],[],[4],[],[]]
        * */
        try {
            val q = LC641.MyCircularDeque(3)
            val r1 = q.insertLast(1)
            val r2 = q.insertLast(2)
            val r3 = q.insertFront(3)
            val r4 = q.insertFront(4)
            val r5 = q.getRear()
            val r6 = q.isFull()
            val r7 = q.deleteLast()
            var r8 = q.insertFront(4)
            var r9 = q.getFront()
            var r10 = q.getRear()
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC641 {
    /// 641. 设计循环双端队列
    class MyCircularDeque(val capacity: Int) {

        val data = IntArray(capacity)
        var start = -1
        var end = -1

        fun insertFront(value: Int): Boolean {
            val size = size()
            if (size == capacity) return false
            if (size == 0) {
                start = 0
                end = 0
            } else {
                start = (start - 1 + capacity) % capacity
            }
            data[start] = value

            return true
        }

        fun insertLast(value: Int): Boolean {
            val size = size()
            if (size == capacity) return false
            if (size == 0) {
                start = 0
                end = 0
            } else {
                end = (end + 1) % capacity
            }
            data[end] = value

            return true
        }

        fun deleteFront(): Boolean {
            if (size() == 0) return false
            if (start == end) {
                start = -1
                end = -1
            } else {
                start = (start + 1) % capacity
            }
            return true
        }

        fun deleteLast(): Boolean {
            if (size() == 0) return false
            if (start == end) {
                start = -1
                end = -1
            } else {
                end = (end - 1 + capacity) % capacity
            }
            return true
        }

        fun getFront(): Int {
            if (size() == 0) return -1
            return data[start]
        }

        fun getRear(): Int {
            if (size() == 0) return -1
            return data[end]
        }

        fun isEmpty(): Boolean {
            return size() == 0
        }

        fun isFull(): Boolean {
            return size() == capacity
        }

        fun size(): Int {
            if (start == -1 || end == -1) return 0
            if (start <= end) return end - start + 1
            return (end + 1) + (capacity - start)
        }

    }

    /**
     * Your MyCircularDeque object will be instantiated and called as such:
     * var obj = MyCircularDeque(k)
     * var param_1 = obj.insertFront(value)
     * var param_2 = obj.insertLast(value)
     * var param_3 = obj.deleteFront()
     * var param_4 = obj.deleteLast()
     * var param_5 = obj.getFront()
     * var param_6 = obj.getRear()
     * var param_7 = obj.isEmpty()
     * var param_8 = obj.isFull()
     */
}