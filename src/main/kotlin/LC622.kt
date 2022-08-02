class LC622 {
    /// 622. 设计循环队列
    class MyCircularQueue(var capacity: Int) {

        val storage = IntArray(capacity)
        var head = -1 // 指向队列的头, 第一个元素 (默认为 -1, 代表空队列)
        var tail = -1 // 指向队列的尾, 最后一个元素 (默认为 -1, 代表空队列)

        fun enQueue(value: Int): Boolean {
            if (isFull()) return false

            if (isEmpty()) {
                head = 0
                tail = 0
                storage[tail] = value
                return true
            }

            tail = (tail + 1) % capacity
            storage[tail] = value
            return true
        }

        fun deQueue(): Boolean {
            if (isEmpty()) return false

            if (head == tail) { // size() == 1
                head = -1
                tail = -1
                return true
            }

            head = (head + 1) % capacity
            return true
        }

        fun Front(): Int {
            if (head == -1) return -1
            return storage[head]
        }

        fun Rear(): Int {
            if (tail == -1) return -1
            return storage[tail]
        }

        fun isEmpty(): Boolean {
            return head == -1 || tail == -1
        }

        fun isFull(): Boolean {
            return size() == capacity
        }

        /// 返回队列包含元素个数
        fun size(): Int {
            if (head == -1 || tail == -1) {
                return 0
            }
            if (tail >= head) {
                return tail - head + 1
            }
            // tail < head
            return (capacity - head) + (tail + 1)
        }

    }

    /**
     * Your MyCircularQueue object will be instantiated and called as such:
     * var obj = MyCircularQueue(k)
     * var param_1 = obj.enQueue(value)
     * var param_2 = obj.deQueue()
     * var param_3 = obj.Front()
     * var param_4 = obj.Rear()
     * var param_5 = obj.isEmpty()
     * var param_6 = obj.isFull()
     */
}