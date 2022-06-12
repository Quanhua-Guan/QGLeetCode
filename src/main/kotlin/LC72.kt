class LC72 {
    /// 72. 编辑距离
    class Solution_DP {
        fun minDistance(wordI: String, wordJ: String): Int {
            val lenIMax = wordI.length
            val lenJMax = wordJ.length

            /// transformCount[lenI][lenJ] 代表 wordI[0 until lenI] 转化到 wordJ[0 until lenJ] 所使用的最少操作次数
            /// 所求结果即为 transformCount[lenIMax][lenJMax]
            val transformCount = Array(lenIMax + 1) { IntArray(lenJMax + 1) }

            transformCount[0][0] = 0
            /// 遍历 wordJ 的空串情况, 即 遍历 wordJ 从0开始的子串长度为 0 的情况
            for (lenI in 1..lenIMax) {
                transformCount[lenI][0] = lenI
            }
            /// 遍历 wordI 的空串情况, 即 遍历 wordI 从0开始的子串长度为 0
            for (lenJ in 1..lenJMax) {
                transformCount[0][lenJ] = lenJ
            }

            for (lenI in 1..lenIMax) {     // 遍历 wordI 从0开始的子串长度, 从1开始
                for (lenJ in 1..lenJMax) { // 遍历 wordj 从0开始的子串长度, 从1开始
                    val i = lenI - 1
                    val j = lenJ - 1
                    if (wordI[i] == wordJ[j]) {
                        transformCount[lenI][lenJ] = transformCount[lenI - 1][lenJ - 1]
                    } else {
                        transformCount[lenI][lenJ] = minOf(
                            transformCount[lenI - 1][lenJ],
                            transformCount[lenI][lenJ - 1],
                            transformCount[lenI - 1][lenJ - 1]
                        ) + 1
                    }
                }
            }

            return transformCount[lenIMax][lenJMax]
        }
    }

    /// 记忆化递归解法
    class Solution_Recursive {
        fun minDistance(word1: String, word2: String): Int {
            val mem = Array(word1.length + 1) { IntArray(word2.length + 1) { -1 } }
            return minDistance_(word1, word1.length, word2, word2.length, mem)
        }

        fun minDistance_(
            word1: String,
            len1: Int,
            word2: String,
            len2: Int,
            mem: Array<IntArray>
        ): Int {
            if (len1 == 0 || len2 == 0) return maxOf(len1, len2)

            fun getMemOrElseCalculate(l1: Int, l2: Int): Int {
                var r = mem[l1][l2]
                if (r == -1) {
                    r = minDistance_(word1, l1, word2, l2, mem)
                }
                return r
            }

            if (mem[len1][len2] != -1) {
                return mem[len1][len2]
            }

            val c1 = word1[len1 - 1]
            val c2 = word2[len2 - 1]
            if (c1 == c2) {
                return getMemOrElseCalculate(len1 - 1, len2 - 1).also { mem[len1][len2] = it }
            }

            // 目标：将 word1 通过编辑转化为 word2，可能操作有插入，删除，替换
            return (minOf(
                getMemOrElseCalculate(len1 - 1, len2),
                getMemOrElseCalculate(len1, len2 - 1),
                getMemOrElseCalculate(len1 - 1, len2 - 1)
            ) + 1).also { mem[len1][len2] = it }
        }
    }
}