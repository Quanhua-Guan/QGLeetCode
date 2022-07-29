class LC593 {
    /// 593. 有效的正方形
    class Solution {
        fun validSquare(p1: IntArray, p2: IntArray, p3: IntArray, p4: IntArray): Boolean {
            return (parallel(p1, p2, p4, p3) && parallel(p4, p1, p3, p2) && equalLength(p1, p2, p1, p4) && perpendicular(p1, p2, p1, p4)) ||
                    (parallel(p1, p2, p4, p3) && parallel(p2, p4, p1, p3) && equalLength(p1, p2, p1, p3) && perpendicular(p1, p2, p1, p3)) ||

                    (parallel(p1, p3, p4, p2) && parallel(p3, p2, p1, p4) && equalLength(p1, p3, p1, p4) && perpendicular(p1, p3, p1, p4)) ||
                    (parallel(p1, p3, p4, p2) && parallel(p3, p4, p1, p2) && equalLength(p1, p3, p1, p2) && perpendicular(p1, p3, p1, p2)) ||

                    (parallel(p1, p4, p2, p3) && parallel(p4, p2, p1, p3) && equalLength(p1, p4, p1, p3) && perpendicular(p1, p4, p1, p3)) ||
                    (parallel(p1, p4, p2, p3) && parallel(p4, p3, p1, p2) && equalLength(p1, p4, p1, p2) && perpendicular(p1, p4, p1, p2))
        }

        // p1->p2 与 p3 -> p4 平行.
        fun parallel(p1: IntArray, p2: IntArray, p3: IntArray, p4: IntArray): Boolean {
            var x1 = p1[0]
            var y1 = p1[1]

            var x2 = p2[0]
            var y2 = p2[1]

            var x3 = p3[0]
            var y3 = p3[1]

            var x4 = p4[0]
            var y4 = p4[1]

            // val slop12 = (y2 - y1) / (x2 - x1)
            // val slop34 = (y4 - y3) / (x4 - x3)
            // slop12 == slop34 || slop12 == -slop34
            // (y2 - y1) * (x4 - x3) == (-+) (y4 - y3) * (x2 - x1)
            val a1 = (y2 - y1) * (x4 - x3)
            val a2 = (y4 - y3) * (x2 - x1)
            return a1 == a2 || a1 == -a2
        }

        // p1->p2 与 p3 -> p4 长度相等.
        fun equalLength(p1: IntArray, p2: IntArray, p3: IntArray, p4: IntArray): Boolean {
            var x1 = p1[0]
            var y1 = p1[1]

            var x2 = p2[0]
            var y2 = p2[1]

            var x3 = p3[0]
            var y3 = p3[1]

            var x4 = p4[0]
            var y4 = p4[1]

            val d1 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
            val d2 = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)
            return d1 == d2
        }

        // p1->p2 与 p3 -> p4 垂直
        fun perpendicular(p1: IntArray, p2: IntArray, p3: IntArray, p4: IntArray): Boolean {
            var x1 = p1[0]
            var y1 = p1[1]

            var x2 = p2[0]
            var y2 = p2[1]

            var x3 = p3[0]
            var y3 = p3[1]

            var x4 = p4[0]
            var y4 = p4[1]

            val v12x = x2 - x1
            val v12y = y2 - y1

            val v34x = x4 - x3
            val v34y = y4 - y3

            // 考虑向量长度为0的情况, 如果存在一个向量为0, 此处判断两个向量不垂直.
            return v12x * v34x + v12y * v34y == 0 && !(v12x == 0 && v12y == 0 || v34x == 0 && v34y == 0)
        }
    }
}