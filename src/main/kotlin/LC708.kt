class LC708 {
    /// 708. 循环有序列表的插入
    /**
     * Definition for a Node.
     */
    class Node(var `val`: Int) {
        var next: Node? = null
    }

    class Solution {
        fun insert(head: Node?, insertVal: Int): Node? {
            val newNode = Node(insertVal)

            if (head == null) {
                newNode.next = newNode
                return newNode
            }

            var current = head!!
            // 判断是否所有的节点都具有相同值`val`
            while (current.next != head && current.next!!.`val` == head.`val`) {
                current = current.next!!
            }
            if (current.next == head) {
                // 所有节点都时同一个`val`
                newNode.next = current.next!!
                current.next = newNode
                return head
            }

            while (true) {
                val next = current.next!!
                if (current.`val` > next.`val`) {
                    // 找到起始点 (最小节点) current.next!!
                    // 找到最大值点 current
                    if (insertVal >= current.`val` || insertVal <= next.`val`) {
                        newNode.next = next
                        current.next = newNode
                        break
                    }
                } else { // current.`val` <= next.`val`
                    if (insertVal >= current.`val` && insertVal <= next.`val`) {
                        newNode.next = next
                        current.next = newNode
                        break
                    }
                }

                current = next
            }

            return head
        }
    }
}