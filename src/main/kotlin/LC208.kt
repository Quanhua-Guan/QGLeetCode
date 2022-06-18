class LC208 {
    /// 208. 实现 Trie (前缀树)
    class Trie() {

        val root = Node('#')

        class Node(val value: Char) {
            var next = Array<Node?>(26) { null }
            var count = 0
        }

        fun insert(word: String) {
            var current = root
            for (c in word) {
                val i = c - 'a'
                if (current.next[i] == null) {
                    current.next[i] = Node(c)
                }
                current = current.next[i]!!
            }
            current.count += 1
        }

        fun search(word: String): Boolean {
            var current = root
            for (c in word) {
                val i = c - 'a'
                if (current.next[i] == null) {
                    return false
                }
                current = current.next[i]!!
            }
            return current.count > 0
        }

        fun startsWith(prefix: String): Boolean {
            var current = root
            for (c in prefix) {
                val i = c - 'a'
                if (current.next[i] == null) {
                    return false
                }
                current = current.next[i]!!
            }
            return true
        }

    }

    /**
     * Your Trie object will be instantiated and called as such:
     * var obj = Trie()
     * obj.insert(word)
     * var param_2 = obj.search(word)
     * var param_3 = obj.startsWith(prefix)
     */
}