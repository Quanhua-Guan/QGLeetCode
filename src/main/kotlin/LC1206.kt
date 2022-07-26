import java.util.*

fun main() {
    while (true) {
        try {
            val sl = LC1206.Skiplist()
            println(sl.add(5))
            println(sl.add(5))
            println(sl.erase(5))
            println(sl.erase(5))
            println(sl.erase(5))
            println(sl.search(5))
            println()
        } catch (e: java.lang.Exception) {
            println(e.stackTrace)
        }
        Thread.sleep(800)
    }

}

class LC1206 {
    /// 1206. 设计跳表
    class Skiplist() {

        /*
        索引层根据原始链表
        preroot
            |
           -1 -> 1 ----------------> 5 -> null                (第二层索引链表)
            |    |                   |
           -1 -> 1 ------> 3 ------> 5 ------> 7 -> null      (第一层索引链表)
            |    |         |         |         |
           -1 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> null (原始链表)
        */

        var preroot = Node(-1)

        val rand = Random()

        class Node(var value: Int) {
            var prev: Node? = null
            var next: Node? = null
            var up: Node? = null
            var down: Node? = null
        }

        fun search(target: Int): Boolean {
            return searchNode(target) != null
        }

        fun searchNode(target: Int): Node? {
            var cur: Node? = preroot
            while (cur != null) {
                if (cur.value == target) {
                    return cur
                }

                if (cur.value > target) {
                    return null
                }

                // cur.value < target
                if (cur.next != null) {
                    val next = cur.next!!
                    if (target < next.value) {
                        cur = cur.down
                    } else { // target >= next.value
                        cur = next
                    }
                } else {
                    // 往下遍历
                    cur = cur.down
                }
            }
            return null
        }

        fun add(num: Int) {
            // prev <= num <= post
            val node = Node(num)

            var cur: Node? = preroot
            while (cur != null) {
                // use preroot of `Node(-1)` to make sure `cur.value <= num` always true
                val next = cur.next
                val down = cur.down
                if (next != null) {
                    if (next.value > num) {
                        if (down != null) {
                            cur = down
                        } else {
                            // 已经遍历到了原始链表
                            // 插入元素
                            node.next = next
                            next?.prev = node

                            cur.next = node
                            node.prev = cur

                            buildIndexRandomly(cur, node, 0)
                            break;
                        }
                    } else { // next.value <= num
                        cur = next
                    }
                } else { // next == null
                    if (down != null) {
                        cur = down // 往下搜索
                    } else { // down == null (遍历到原始链表末尾了)
                        // 插入元素
                        node.next = next
                        next?.prev = node

                        cur.next = node
                        node.prev = cur

                        buildIndexRandomly(cur, node, 0)
                        break;
                    }
                }
            }
        }

        fun buildIndexRandomly(cur: Node, inserted: Node, level: Int) {
            if (rand.nextInt(Math.pow(2.0, level.toDouble()).toInt()) != 0 || level == 16) {
                return
            }

            // 从后往前建索引链表
            var cur = cur
            while (cur != null) {
                // 检查cur是否包含up节点 (如果有up节点, 则说明 cur.up.prev 是存在的)
                if (cur.up != null) {
                    val curUp = cur.up!!
                    val curUpNext = curUp.next

                    val insertedUp = Node(inserted.value)
                    insertedUp.down = inserted
                    inserted.up = insertedUp

                    insertedUp.next = curUpNext
                    curUpNext?.prev = insertedUp

                    curUp.next = insertedUp
                    insertedUp.prev = curUp

                    buildIndexRandomly(curUp, insertedUp, level + 1)
                    return;
                }

                // 检测cur是否有前一个节点 cur.prev != null
                if (cur.prev == null) {
                    // 没有前一个节点, 说明已经遍历到头结点了, 需要主动为其创建一个[上节点]
                    val curUp = Node(cur.value)
                    cur.up = curUp
                    curUp.down = cur

                    preroot = curUp
                    continue // 跳到下次循环
                }

                // 将cur替换为cur的前一个节点
                cur = cur.prev!!
            }
        }

        fun erase(num: Int): Boolean {
            assert(num >= 0)
            var cur = searchNode(num)
            if (cur == null) return false

            while (cur != null) {
                val prev = cur.prev!!
                prev.next = cur.next
                cur.next?.prev = prev

                cur = cur.down
            }

            return true
        }

    }

    /**
     * Your Skiplist object will be instantiated and called as such:
     * var obj = Skiplist()
     * var param_1 = obj.search(target)
     * obj.add(num)
     * var param_3 = obj.erase(num)
     */
}