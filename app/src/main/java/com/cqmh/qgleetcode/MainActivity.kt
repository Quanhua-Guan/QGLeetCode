package com.cqmh.qgleetcode

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import java.lang.Exception
import java.math.BigInteger
import java.util.*
import kotlin.collections.ArrayList

// 两数之和
class Solution1 {
    fun twoSum(nums: IntArray, target: Int): IntArray {
        var numMap = mutableMapOf<Int, Int>()
        var index = 0
        var result = intArrayOf(0, 0)
        for (num in nums) {
            if (numMap.containsKey(target - num)) {
                result[0] = numMap[target - num]!!
                result[1] = index
                break;
            } else {
                numMap[num] = index
            }
            index++
        }

        return result;
    }

    fun twoSum0(nums: IntArray, target: Int): IntArray {
        var numMap = mutableMapOf<Int, MutableList<Int>>()
        var index = 0
        for (num in nums) {
            if (!numMap.containsKey(num)) {
                var list = mutableListOf<Int>()
                list.add(index)
                numMap[num] = list
            } else {
                var list = numMap[num]!!
                list.add(index)
            }
            index++
        }

        val result = IntArray(2)
        for (num in nums) {
            val other = target - num
            if (numMap.containsKey(num) && numMap.containsKey(other)) {
                val numList = numMap[num]!!
                if (num == other) {
                    if (numList.size >= 2) {
                        result[0] = numList[0]
                        result[1] = numList[1]
                        break;
                    }
                } else {
                    val otherList = numMap[other]!!
                    result[0] = numList[0]
                    result[1] = otherList[0]
                }
            }
        }

        return result;
    }
}

class ListNode(var `val`: Int) {
    var next: ListNode? = null
}

// 两数相加
class Solution2 {
    fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
        var l1 = l1
        var l2 = l2
        var carry = 0 // 进位
        var result = ListNode(-1)
        var current = result

        // 1, 2, 3
        // 2, 3,

        // 1, 2, 3
        // 1, 2, 3

        // 1, 2
        // 1, 2, 3

        while (l1 != null || l2 != null) {
            val val1 = l1?.`val` ?: 0
            var val2 = l2?.`val` ?: 0

            val sum = val1 + val2 + carry
            carry = sum / 10

            current.next = ListNode(sum % 10)
            current = current.next!!

            l1 = l1?.next
            l2 = l2?.next
        }

        if (carry > 0) {
            current.next = ListNode(carry)
        }

        return result.next
    }
}

class Solution9 {
    fun isPalindrome(x: Int): Boolean {
        if (x < 0) return false

        val nums = mutableListOf<Int>()
        var x = x
        while (x != 0) {
            nums.add(x % 10)
            x /= 10
        }

        var l = 0
        var r = nums.size - 1
        while (l < r) {
            if (nums[l] != nums[r]) return false
            l++
            r--
        }

        return true
    }

    fun isPalindrome1(x: Int): Boolean {
        if (x < 0) return false

        // 121 => 12 => 1 => 0
        // 1   => 12 => 121 =>x

        var xx = x
        var invertX = 0
        while (xx != 0) {
            invertX = invertX * 10 + (xx % 10)
            xx /= 10
        }

        return x == invertX
    }
}

class Solution3 {
    fun lengthOfLongestSubstring(s: String): Int {
        // 3个关键词
        // 最长, 记录 max
        // 无重复, 用字典记录是否已存在
        // 子串, 自测必然有 start 和 end 两个下标
        var charIndexMap = mutableMapOf<Char, Int>()
        var maxLength = 0
        var start = 0
        var current = 0

        while (current < s.length) {
            val char = s[current]
            if (charIndexMap.containsKey(char) && charIndexMap[char]!! >= start) {
                start = charIndexMap[char]!! + 1
            } else {
                maxLength = maxOf(maxLength, current - start + 1)
            }
            charIndexMap[char] = current
            current++
        }

        return maxLength
    }
}

class Solution647 {
    // 回文子串数
    /**
    给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
    回文字符串 是正着读和倒过来读一样的字符串。
    子字符串 是字符串中的由连续字符组成的一个序列。
    具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
     */

    ///// 暴力解法
    fun countSubstrings(s: String): Int {
        var count = 0
        for (i in s.indices) {
            for (j in i + 1 until s.length) {
                val substring = s.substring(i, j)
                if (isPalindrome(substring)) {
                    count += 1
                }
            }
        }
        return count
    }

    fun isPalindrome(s: String): Boolean {
        var l = 0
        var r = s.length - 1
        while (l < r) {
            if (s[l] != s[r]) {
                return false
            }
            l++
            r--
        }
        return true
    }

    ///// 中心扩散法

    fun countSubstrings1(s: String): Int {
        var count = 0
        for (i in s.indices) {
            count += palindromeSubstringAtCenter(s, i)
        }
        return count
    }

    fun palindromeSubstringAtCenter(s: String, index: Int): Int {
        var count = 0

        // 以当前下标为中心的子串 (基数长度)
        var l = index
        var r = index
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            count++
            l--
            r++
        }

        // 以当前下标为中左的子串 (偶数长度)
        l = index
        r = index + 1
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            count++
            l--
            r++
        }

        return count
    }
}

/// 908. 最小差值 I
class Solution908 {
    // 思考: 读懂题意很重要, 是重要的第一步.
    // 计算两个数之间差值的最小值
    fun smallestRangeI(nums: IntArray, k: Int): Int {
        var min = Int.MAX_VALUE
        var max = Int.MIN_VALUE
        for (i in nums.indices) {
            min = minOf(min, nums[i])
            max = maxOf(max, nums[i])
        }

        if (max - min <= 2 * k) {
            return 0
        }
        return max - min - 2 * k
    }
}

/// 最长回文子串
class Solution5 {

    ///// 中心扩散法

    fun longestPalindrome(s: String): String {
        var longestLength = 0
        var center = -1
        for (i in s.indices) {
            val length = longestPalindromeLengthWithCenter(s, i)
            if (length > longestLength) {
                longestLength = length
                center = i
            }
        }

        val step = longestLength / 2
        if (longestLength % 2 == 1) {
            return s.substring(center - step, center + step + 1)
        }
        return s.substring(center - step + 1, center + step + 1)
    }

    fun longestPalindromeLengthWithCenter(s: String, center: Int): Int {
        if (s.isEmpty()) return 0

        var max = 0
        var l = center
        var r = center
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            max = maxOf(max, r - l + 1)
            l--
            r++
        }

        l = center
        r = center + 1
        while (l >= 0 && r < s.length && s[l] == s[r]) {
            max = maxOf(max, r - l + 1)
            l--
            r++
        }

        return max
    }

    ///// 动态规划

    // dp[i][j] = s[i] == s[j] && ((j - 1) - (i + 1) < 1 || dp[i+1][j-1])
    // dp[i][j] = s[i] == s[j] && (j - i < 3 || dp[i+1][j-1])
    // i < j
    fun longestPalindrome_dp(s: String): String {
        if (s.isEmpty()) return ""

        val dp = Array(s.length) { BooleanArray(s.length) }

        var maxLength = 0
        var start = 0

        for (j in 1 until s.length) {
            for (i in 0 until j) {
                if (s[i] != s[j]) {
                    dp[i][j] = false
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true
                    } else {
                        dp[i][j] = dp[i + 1][j - 1]
                    }
                }

                if (dp[i][j] && j - i + 1 > maxLength) {
                    maxLength = j - i + 1
                    start = i
                }
            }
        }

        return s.substring(start, start + maxLength)
    }
}

// 整数反转
class Solution7 {
    fun reverse(x: Int): Int {
        if (x == Int.MIN_VALUE) return 0

        val isNagtive = x < 0
        var x = if (isNagtive) -x else x
        var result = 0
        while (x != 0) {
            if ((result * 10 + (x % 10) - (x % 10)) / 10 != result) {
                return 0 // 注意溢出问题
            }
            result = result * 10 + (x % 10)
            x /= 10
        }

        return if (isNagtive) -result else result
    }
}

// 8. 字符串转换整数 (atoi)
class Solution8 {
    fun myAtoi(s: String): Int {
        var i = 0
        while (i < s.length && s[i] == ' ') {
            i++
        }

        var sign = 1
        if (i < s.length) {
            if (s[i] == '+') {
                i++
            } else if (s[i] == '-') {
                sign = -1
                i++
            }
        }

        var result = 0
        if (i < s.length && (s[i] in '0'..'9')) {
            result = (s[i] - '0') * sign
            i++
        }

        while (i < s.length && (s[i] in '0'..'9')) {
            val digit = (s[i] - '0') * sign
            if (result * 10 / 10 != result || (result < 0 && result * 10 + digit > 0) || (result > 0 && result * 10 + digit < 0)) {
                // 越界了
                return if (sign == 1) MAX_VALUE else MIN_VALUE
            }
            result = result * 10 + digit

            i++
        }

        return result
    }

    companion object {
        public const val MIN_VALUE: Int = -2147483648
        public const val MAX_VALUE: Int = 2147483647
    }
}

// Z 字形变换
class Solution6 {
    fun convert(s: String, numRows: Int): String {
        if (s.length <= 1 || numRows == 1) return s

        var charLists = List(numRows) { LinkedList<Char>() }

        // if numRows == 4
        // 0, 1, 2, 3, 2, 1
        var index = 0
        var ascend = true
        var i = 0
        while (i < s.length) {
            val char = s[i]
            charLists[index].add(char)

            if (ascend) {
                if (index == numRows - 1) {
                    ascend = false
                    index--
                } else {
                    index++
                }
            } else {
                if (index == 0) {
                    ascend = true
                    index++
                } else {
                    index--
                }
            }

            i++
        }

        var s = StringBuffer()
        for (charList in charLists) {
            charList.forEach { s.append(it) }
        }

        return s.toString()
    }
}

/// 4. 寻找两个正序数组的中位数
class Solution4 {
    fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
        val totalSize = nums1.size + nums2.size
        if (totalSize % 2 == 1) { // 总共 3 个数, 则 3 / 2 = 1, 应取第 3 / 2 + 1 个数
            return findNth(nums1, nums2, totalSize / 2 + 1)
        }
        // 总共 4 个数, 则 4 / 2 = 2, 应取第 2 和第 3 个数求平均值
        return (findNth(nums1, nums2, totalSize / 2) + findNth(nums1, nums2, totalSize / 2 + 1)) / 2
    }

    /// 查找两个有序数组的第 N 小的数
    fun findNth(nums1: IntArray, nums2: IntArray, n: Int): Double {
        var start1 = 0
        var start2 = 0
        /// 每次从 nums1 和 nums2 中, 以 start0 和 start1 为起点, 分别最多选出前 n/2 个数字构成 tmpNums1, tmpNums2, 则总共 选出了 n 个数字, 从 tmpNums1 和 tmpNums2 中取最大数字v1, v2(即最后一个数),
        /// 比较 v1, v2
        /// - 如果 v1 == v2, 返回 v1
        /// - 如果 v1 < v2, 则 v2 在所有数字中最高排名为 n, v2 在所有数字钟最高排名为 n - 1, 说明 tmpNums1 中的所有数字必然不可能为第 n 个数, 所以可以排除. 然后更新 start1, n.
        /// - 如果 v1 > v2, 同理, 可以将 tmpNums2 中的所有数字排除. 然后更新 start1, n.
        var n = n

        while (n > 1) {
            // 取值前先算下标, 不能越界
            // nums1, 从 start1 开始(包括 start1)依次选择最多 n/2 个数字 (最少可能选择 0 个数)
            val i1 = if (start1 >= nums1.size) start1 else minOf(start1 + n / 2 - 1, nums1.size - 1)
            // nums2, 同理
            val i2 = if (start2 >= nums2.size) start2 else minOf(start2 + n / 2 - 1, nums2.size - 1)

            // 取值
            val v1 = nums1.getOrElse(i1) { Int.MAX_VALUE }
            val v2 = nums2.getOrElse(i2) { Int.MAX_VALUE }

            // 判断
            if (v1 < v2) {
                n -= i1 - start1 + 1
                start1 = i1 + 1
            } else {
                n -= i2 - start2 + 1
                start2 = i2 + 1
            }
        }

        // 此时 n == 1
        val v1 = nums1.getOrElse(start1) { Int.MAX_VALUE }
        val v2 = nums2.getOrElse(start2) { Int.MAX_VALUE }
        return minOf(v1, v2).toDouble()
    }
}

/// 392. 判断子序列
class Solution392 {
    // 判断字符串 s 是否是字符串 t 的子序列
    fun isSubsequence(s: String, t: String): Boolean {
        if (s.length > t.length) return false
        if (s.isEmpty()) return true

        var iS = 0
        var iT = 0
        while (iS < s.length && iT < t.length) {
            var valS = s[iS]
            var valT = valS - 1
            while (iT < t.length) {
                valT = t[iT]
                if (valS == valT) {
                    break
                }
                iT++
            }

            if (valS != valT) {
                return false
            }

            iS++
            iT++
        }

        return iS == s.length
    }
}

//  188. 买卖股票的最佳时机 IV
class Solution188 {
    fun maxProfit(k: Int, prices: IntArray): Int {
        if (prices.isEmpty()) return 0

        // dp[j][i] 在第 i 天最多操作 j 次的最大利润
        // k + 1 记录操作 0, 1, 2, ...k 次 (共 k+1种情况)
        // prices.size 天, 第 0, 1, 2, ...prices.size-1 天
        var dp = Array(k + 1) { IntArray(prices.size) }

        for (j in 1 until k + 1) { // 操作 j 次, j == 0时, 默认 dp[j][i] == 0, 所以从 j=1开始
            // 表示手上持有股票的前提下，i-1 天的总收益最大值
            // 第 0 天第一次操作后的收益（第一次操作一定是买入）
            var yestodayMaxWhenHold = -prices[0]
            for (i in 1 until prices.size) {
                // 第 i 天，第一天(i==0)只可能 1）什么都不做， 2）买入， 第二天开始出了以上两项， 增加操作 3）卖出
                //  所以第一天 dp[j][0] 最优质都为0， 所以 i 从 1 开始遍历（即从第二天开始）。
                dp[j][i] = maxOf(
                    //  第 i 天躺平，什么都不做
                    dp[j][i - 1],
                    //  第 i 天卖出，能卖出则说明已买入，max记录第 i-1 天及之前的最大收益，在此基础上收益 prices[i]。
                    // 比如：第二天卖出， 则第一天必然买入（参考 max 的初始值设置）
                    prices[i] + yestodayMaxWhenHold
                )
                // dp[j][i]为最大利润时，必然已经将股票卖出了或第i 天股票价格为0（因为股票价格总是为非负数）
                //
                // 相隔的两天的操作一定不可能为『买入买入』或『卖出卖出』，两次『买入』之间必须间隔至少一次『卖出』
                //
                // 为下次迭代准备（为下次卖出场景准备），更新 yestodayMaxWhenHold 为表示手上持有股票的前提下，i 天的总收益最大值，可能情况：
                // 1）第 i 天不买入，则必须要求i-1天或之前已买买入，即 yestodayMaxWhenHold
                // 2）第 i 天买入， 则在第i-1天的最大利润基础上，买入股票，即减去第 i 天的股票价格（prices[i]）
                // 比较取更大的值
                yestodayMaxWhenHold = maxOf(yestodayMaxWhenHold, dp[j - 1][i - 1] - prices[i])
            }
        }

        return dp[k][prices.size - 1]
    }

    fun maxProfit1(k: Int, prices: IntArray): Int {
        // 从题目示例可以得出：连续的一次买入和一次卖出为一次交易
        if (k < 1) {
            return 0 // 不能买不能卖，最好的策略不买，利润最高0
        }

        // 因为当天买入候随即卖出不会改变当天的利润，所以不考虑这种操作

        // dayCount 天数，等于 prices.size
        val dayCount = prices.size
        if (dayCount <= 1) {
            return 0 // 最好的策略是不买，或者买入当天卖出，利润最高0
        }

        // trasactionCount 交易次数：0，1，2...k+1
        // day 日期：0，1，2...dayCount-1
        // maxProfit[transactionCount][day] 第 day 天最多交易 transactionCount 次的最大利润
        var maxProfit = Array(k + 1) { IntArray(dayCount) }

        // 在买入的情况下第最大的利润
        var maxProfitWhenHold = 0

        // 交易0次，最大利润为0。进行最多1次交易，最大利润为0，已买入的情况下最大利润为 -prices[0]
        // 所以，交易次数从1开始
        for (transactionCount in 1 until k + 1) {
            maxProfitWhenHold = -prices[0]
            // 第0天最大利润为0，所以从第1天开始
            // 每天能执行的操作只有3种：
            // 1）买入
            // 2）卖出
            // 3）不买不卖（或买入即卖出）
            for (day in 1 until dayCount) {
                maxProfit[transactionCount][day] =
                    maxOf(maxProfit[transactionCount][day - 1], prices[day] + maxProfitWhenHold)
                maxProfitWhenHold =
                    maxOf(maxProfitWhenHold, maxProfit[transactionCount - 1][day - 1] - prices[day])
            }
        }

        return maxProfit[k][dayCount - 1]
    }
}

/// 70. 爬楼梯
class Solution70 {
    fun numWays(n: Int): Int {
        // 跳法 ways[n]=ways[n-1]+ways[n-2]
        // 台阶编号 0（代表地面），1，2，3...，n，显然 ways[0]=0, ways[1]=1
        var ways = IntArray(n + 1)
        if (n == 0) return 0
        if (n == 1) return 1
        if (n == 2) return 2
        ways[1] = 1
        ways[2] = 2
        for (i in 3 until n + 1) {
            ways[i] = ways[i - 1] + ways[i - 2]
        }
        return ways[n]
    }
}

// 62. 不同路径
class Solution62 {
    fun uniquePaths(m: Int, n: Int): Int {
        // 到达坐标点点总路径数
        // paths[i][j] = paths[i-1][j] + paths[i][j-1]
        var paths = Array(m) { IntArray(n) }
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (i == 0 || j == 0) {
                    paths[i][j] = 1
                } else {
                    paths[i][j] = paths[i - 1][j] + paths[i][j - 1]
                }
            }
        }

        return paths[m - 1][n - 1]
    }
}

/// 64. 最小路径和
class Solution {
    fun minPathSum(grid: Array<IntArray>): Int {
        // m 行，n 列，grid[i][j]为非负整数（0<=i<m, 0<=j<n）
        val m = grid.size
        val n = grid.getOrElse(0) { IntArray(0) }.size

        if (m == 0 || n == 0) return 0

        // 路径总和最小值
        // cost[i][j] = maxOf(coast[i-1][j], coast[i][j-1]) + grid[i][j])
        var cost = Array(m) { IntArray(n) }

        // 初始化第一行第一列为 grid[0][0]
        cost[0][0] = grid[0][0]
        // 第一列 从第二行开始，从上往下
        for (i in 1 until m) {
            cost[i][0] = cost[i - 1][0] + grid[i][0]
        }
        // 第一行 从第二列开始，从左往右
        for (j in 1 until n) {
            cost[0][j] = cost[0][j - 1] + grid[0][j]
        }
        // 从第二行第二列开始遍历
        for (i in 1 until m) {
            for (j in 1 until n) {
                cost[i][j] = minOf(cost[i - 1][j], cost[i][j - 1]) + grid[i][j]
            }
        }
        return cost[m - 1][n - 1]
    }
}

/// 72. 编辑距离
class Solution72 {
    fun minDistance(word1: String, word2: String): Int {
        val l1 = word1.length + 1
        val l2 = word2.length + 1

        /// 例子： house => ros
        ///       j
        ///       0 1 2 3
        ///       # r o s
        /// i 0 # 0 1 2 3 => 从空字符串 a 转化到非空字符串 b，编辑距离为 b.length
        ///   1 h 1
        ///   2 o 2
        ///   3 u 3
        ///   4 s 4
        ///   5 e 5
        ///       从非空字符串 b 转化到空字符串 a，编辑距离为 b.length

        /// op[i][j] 代表 word1[0..<i] 转化到 word2[0..<j] 所使用的最少操作次数
        /// word1[0..<i] 代表 word1 中从第 1 个字符开始到第 i 个字符结束（包含第 i 个字符）构成的子字符串, 例如：word1[0..<word1.length] 即为 word1.
        val op = Array(l1) { IntArray(l2) }

        /// 可能操作有3种，插入，删除，替换。一次操作包括操作类型和操作位置。同一组操作按不同顺序执行，得到的结果是一样的。
        /// 所以将问题简化，从 word1 和 word2 的最后一个字符往前推导并进行操作：
        /// 设 i 指向 word1 的最后一个字符，设 j 指向 word2 的最后一个字符。
        ///（1）如果 word1[i] == word2[j]，则 op[i][j] = op[i-1][j-1] (相当于直接将 word1 和 word2 的最后一个字符删除，因为相同所以不需要转化，可以不考虑)
        ///（2）如果 word1[i] ！= word2[j]，则对 word1[0..<i] 可以进行 3 种操作，以保证 word1[0..<i] 转化为 word2[0..<j]，如下：
        ///     1）将字符 word2[j-1] 插入 word1[0..<i] 末尾。相当于先将 word1[0..<i] 转化为 word2[0..<j-1]，所需步数为 op[i][j-1], 然后在末尾插入字符 word2[j-1]，转化为 word2[0..<j]，所需步数为 1。总结一下：
        ///        op[i][j]=op[i][j-1]+1
        ///     2）将字符 word1[i-1] 删除，即 word1[0..<i] 转化为 word1[0..<i-1]，所需步数 1，然后将 word1[0..<i-1] 转化为 word2[0..<j], 所需步数为 op[i-1][j]。总结一下：
        ///        op[i][j]=1+op[i-1][j]
        ///     3）将字符 word1[i-1] 替换成 word2[j-1]，相当于先做字符替换，步数为1，替换后 word1[i-1] 和 word2[j-1] 相等，所以直接忽略，然后将 word1[0..<i-1] 转化为 word2[0..<j-1]，所需步数为 op[i-1][j-1]。总结一下：
        ///        op[i][j]=1+op[i-1][j-1]
        ///     从以上3种情况种选出最小值即为 op[i][j]（最少操作步数）

        for (i in 1 until l1) { // i 代表 word1 的长度，例如：i为1则对应 word1[0..<1], i为2则对应 word2[0..<2]
            op[i][0] = op[i - 1][0] + 1
        }

        for (j in 1 until l2) {
            op[0][j] = op[0][j - 1] + 1
        }

        for (i in 1 until l1) {
            for (j in 1 until l2) {
                if (word1[i - 1] == word2[j - 1]) {
                    op[i][j] = op[i - 1][j - 1]
                } else {
                    op[i][j] = minOf(
                        op[i][j - 1] + 1,
                        op[i - 1][j] + 1,
                        op[i - 1][j - 1] + 1
                    )
                }
            }
        }

        return op[l1 - 1][l2 - 1]
    }

    //// 递归解法 （超时了）

    fun minDistance1(word1: String, word2: String): Int {
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
        if (len1 == 0 || len2 == 0) return maxOf(len1, len2).also { mem[len1][len2] = it }

        val c1 = word1[len1 - 1]
        val c2 = word2[len2 - 1]

        fun getMemOrElseCalculate(l1: Int, l2: Int): Int {
            var r = mem[l1][l2]
            if (r == -1) {
                r = minDistance_(word1, l1, word2, l2, mem)
            }
            return r
        }

        if (c1 == c2) {
            return getMemOrElseCalculate(len1 - 1, len2 - 1)
        }

        // 目标：将 word1 通过编辑转化为 word2，可能操作有插入，删除，替换
        return minOf(
            getMemOrElseCalculate(len1 - 1, len2),
            getMemOrElseCalculate(len1, len2 - 1),
            getMemOrElseCalculate(len1 - 1, len2 - 1)
        ) + 1

    }
}

// 10. 正则表达式匹配
class Solution10 {
    ///  匹配 "*"  和 "."
    fun isMatch(s: String, p: String): Boolean {
        // 给 p 分词，从后往前扫描，
        // 1）如果遇到"*"则将"*"和它的前一个字符打包作为一个匹配项。
        // 2）否则必然遇到的是"a-z"或"."
        // 匹配项数组
        var patterns = ArrayList<String>()
        var i = 0
        while (i < p.length) {
            val hasNext = i + 1 < p.length
            if (hasNext && p[i + 1] == '*') {
                patterns.add("" + p[i] + p[i + 1])
                i++
            } else {
                patterns.add(p[i].toString())
            }
            i++
        }

        // 原字符串前 i 个字符构成的子串是否和匹配项数组前 j 个匹配项组成的匹配字符串匹配
        // +1是因为需要预留一个空字符和空匹配项
        val matches = Array(s.length + 1) { BooleanArray(patterns.size + 1) }

        // 默认空串和空串匹配项匹配
        matches[0][0] = true
        /* // 默认为 false, 可不处理
        for (i in 1 until s.length + 1) {
            matchs[i][0] = false
        }
         */
        // 空字符串和匹配项进行匹配
        for (j in 1 until patterns.size + 1) {
            matches[0][j] = patterns[j - 1].last() == '*' && matches[0][j - 1]
        }

        for (i in 1 until s.length + 1) {
            for (j in 1 until patterns.size + 1) {
                matches[i][j] =
                        //  字符串 s[0..<i-1] 和 patterns[0..<j] 匹配，且当前字符和当前匹配项匹配，并且匹配项时通配符
                    (matches[i - 1][j] && isMatchEmpty(patterns[j - 1]) && isCharMatch(
                        s[i - 1],
                        patterns[j - 1]
                    )) ||
                            // 字符串 s[0..<i-1] 和 patterns[0..<j-1] 匹配，且且当前字符和当前匹配项匹配
                            (matches[i - 1][j - 1] && isCharMatch(s[i - 1], patterns[j - 1])) ||
                            // 字符串 s[0..<i] 和 patterns[0..<j-1] 匹配， 且当前匹配项时通配符（可以匹配空字符串）
                            (matches[i][j - 1] && isMatchEmpty(patterns[j - 1]))
            }
        }

        return matches[s.length][patterns.size]
    }

    // 字符是否和匹配项匹配，匹配项可能是 a-z 或 . 中的一个字符，或者是 a-z 或 . 中的一个字符加一个*, 例如："a*", ".*"
    private fun isCharMatch(c: Char, p: String): Boolean {
        return (p.length == 1 && (c == p[0] || p[0] == '.')) ||
                (p.length == 2 && p[1] == '*' && (c == p[0] || p[0] == '.'))
    }

    private fun isMatchEmpty(p: String): Boolean {
        return p.length == 2 && p[1] == '*'
    }
}

/// 96. 不同的二叉搜索树
/// 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树（BST）的种数。
class Solution96 {
    fun numTrees(n: Int): Int {
        /*
        思路：
        （1）1 到 n 组成的 n 个数顺序数组，从中任选一个数 i 为根，其左边的数为左子树（有 i-1 个数，假设有 a 种），右边的数为右子树（有 n-i 个数，假设有 b 种）。依次类推对左子树和右子树做同样的操作。
            1）所以，以 i 为根的 BST 一共有 a*b 种。
            2）i 的选择可能有 n 种。
        （2）设 counts[i] (1 <= i <= n) 为 i 个数顺序数组构成的二叉搜索树的种数。 count[n] 即为我们所求的结果。
        （3）设 countOfRoot[i, n] 为以数字 i 为根，节点个数为 n 的 BST 的种数，则 countOfRoot[i, n] = counts[i-1] * counts[n-i]
        （4）i 分别取 1 到 n，计算 countOfRoot[i, n]，然后将所有 countOfRoot[i, n] 加和即为 counts[n]，根据（3）可以得出，
            i 分别取 1 到 n，计算 counts[i-1] * counts[n-i]，所有加和得到结果 counts[n]
        （5）counts[0] = 1, counts[1] = 1
        */
        var counts = IntArray(n + 1)
        counts[0] = 1
        counts[1] = 1
        for (c in 2 until n + 1) {// 计算 c 从 2 到 n，c 为 count 缩写。
            for (r in 1 until c + 1) {// 选 BST 的根，r 为 root 缩写。
                counts[c] += counts[r - 1] * counts[c - r]
            }
        }
        return counts[n]
    }

    /// 卡塔兰数
    fun numTrees1(n: Int): Int {
        var c = 1L
        // c(0) = 1, c(n+1)=c(n)*(2(2n+1)/(n+2))
        for (k in 0 until n) {
            c = c * 2 * (2 * k + 1) / (k + 2)
        }
        return c.toInt();
    }

    fun numTrees2(n: Int): Int {
        // 根据 n 的取值分析结果，观察规律
        // c(0) = 1
        // c(1) = 1
        // c(2) = 2
        // c(3) = 5
        // 符合卡特兰数列规律
        // c(n) = c(n-1)*(4*n-2)/(n+1)
        var c = 1L // c(0)
        for (i in 1 until n + 1) {
            // 求 c(i)
            c = c * (4 * i - 2) / (i + 1)
        }

        // 此时 i 等于 n+1，c 承载 c(n) 的结果

        return c.toInt()
    }
}

/// 946. 验证栈序列
/*
 给定pushed和popped两个序列，每个序列中的 值都不重复，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false。
 */
class Solution946 {
    // 思路：模拟进出栈操作
    fun validateStackSequences(pushed: IntArray, popped: IntArray): Boolean {
        var stack = LinkedList<Int>()
        var pushedIndex = 0
        var poppedIndex = 0
        while ((pushedIndex < pushed.size || stack.size > 0) && poppedIndex < popped.size) {
            val top = stack.lastOrNull()
            val pus = pushed.getOrNull(pushedIndex)
            val pop = popped[poppedIndex]
            if (pop == pus) {
                pushedIndex++
                poppedIndex++
            } else if (pop == top) {
                stack.removeLast()
                poppedIndex++
            } else {
                // pop != top && pop != pus
                if (pus != null) {
                    stack.add(pus)
                    pushedIndex++
                } else {
                    return false
                }
            }
        }
        return pushedIndex == pushed.size && stack.size == 0 && poppedIndex == popped.size
    }
}

/// 957. N 天后的牢房
class Solution957 {
    /// 方法1：暴力模拟，超时！！！
    fun prisonAfterNDays(cells: IntArray, n: Int): IntArray {
        // f(n,i) 代表第 n 天第 i 个房间是否有人, n 代表天数（0,1,2,...N-1）,i代表房间号（0，1，2，3，4，5，6，7）
        // f(0,i) = cells[i] （初始值）

        // (1) f(n,0) = 0, if n > 0 第一个房间从第2天开始总是空置，第8个房间同理，f(n,7) = 0, if n > 0
        // (2) f(n,i) = if f(n-1, i-1) == f(n-1, i+1) 1 else 0

        var cells = cells
        // 需要一个临时空间暂存昨天的房间状态
        var yestodayCells = cells.clone()

        for (n in 1 until n + 1) {
            cells[0] = 0
            cells[cells.size - 1] = 0
            for (i in 1 until cells.size - 1) {
                cells[i] = if (yestodayCells[i - 1] == yestodayCells[i + 1]) 1 else 0
            }
            yestodayCells = cells.also { cells = yestodayCells }
        }

        return yestodayCells
    }

    /// 方法2：将数组换成 Int
    /// 假设 cells 共 8 位，则对应的 Int 取值总共有 2^8 = 256 种取值
    fun prisonAfterNDays1(cells: IntArray, n: Int): IntArray {
        // f(n,i) 代表第 n 天第 i 个房间是否有人, n 代表天数（0,1,2,...N-1）,i代表房间号（0，1，2，3，4，5，6，7）
        // f(0,i) = cells[i] （初始值）

        // (1) f(n,0) = 0, if n > 0 第一个房间从第2天开始总是空置，第8个房间同理，f(n,7) = 0, if n > 0
        // (2) f(n,i) = if f(n-1, i-1) == f(n-1, i+1) 1 else 0

        // 返回 1 或 0
        fun getBit(value: Int, bitIndex: Int): Int {
            return (value and (1 shl (bitIndex - 1))) shr (bitIndex - 1)
        }

        // bitValue = 0 or 1
        fun valueWithNewBit(value: Int, bitIndex: Int, bitValue: Int): Int {
            return value or (bitValue shl (bitIndex - 1))
        }

        // 第 0 天的状态
        var rooms = 0
        for (i in cells.indices) {
            rooms = rooms or (cells[i] shl (cells.size - i - 1))
        }

        // rooms 状态 => 天数
        var history = mutableMapOf<Int, Int>()
        var day = 1
        var didJump = false
        while (day <= n) {
            val yestodayRooms = rooms

            if (!didJump) {
                if (history.containsKey(rooms)) {
                    val historyDay = history[rooms]!!
                    val period = day - historyDay
                    day += (n - day) / period * period // 跨过所有中间循环周期，跳到最后一次循环周期
                    didJump = true
                } else {
                    history[rooms] = day
                }
            }

            // 第一位和最后一位清零
            rooms = 0
            // 给房间重新编号 (和 Int 二进制位对应)
            // 只需遍历第 2 到第 7 位，第 1 和第 8 位已经重置为 0
            // 8 - 7 - 6 - 5 - 4 - 3 - 2 - 1
            for (i in 2 until cells.size) {
                val l = getBit(yestodayRooms, i - 1)
                val r = getBit(yestodayRooms, i + 1)
                val bitValue = if (l == r) 1 else 0
                rooms = valueWithNewBit(rooms, i, bitValue)
            }

            day++
        }

        for (i in cells.indices) {
            val shiftCount = cells.size - i - 1
            cells[i] = (rooms and (1 shl shiftCount)) shr shiftCount
        }

        return cells
    }
}

/// 935. 骑士拨号器
class Solution935 {
    fun knightDialer(n: Int): Int {
        val numberCount = 10
        var numberRelations = listOf<IntArray>(
            intArrayOf(4, 6),    // 4 -> 0, 6 -> 0，可以从 4 或 6 跳到 0
            intArrayOf(6, 8),    // 6 -> 1, 8 -> 1
            intArrayOf(7, 9),    // 7 -> 2, 9 -> 2
            intArrayOf(4, 8),    // 4 -> 3, 8 -> 3
            intArrayOf(0, 3, 9), // 0 -> 4, 3 -> 4, 9 -> 4
            intArrayOf(),        // 无 -> 5
            intArrayOf(0, 1, 7), // 0 -> 6, 1 -> 6, 7 -> 6
            intArrayOf(2, 6),    // 2 -> 7, 6 -> 7
            intArrayOf(1, 3),    // 1 -> 8, 3 -> 8
            intArrayOf(2, 4)    // 2 -> 9, 4 -> 9
        )

        // 跳 n 次后落到数字 i 的总共电话号数目
        // counts[i][n] = counts[numberRelations[i][0]][n - 1] + counts[numberRelations[i][1]][n - 1] + ...
        var counts = Array(numberCount) { Array<BigInteger>(n + 1) { BigInteger.valueOf(0) } }

        // 初始化
        for (i in 0 until numberCount) {
            counts[i][1] = BigInteger.valueOf(1)
        }

        // 跳 n 次
        for (jump in 2 until n + 1) {
            for (number in 0 until numberCount) {
                var sum = BigInteger.valueOf(0)
                numberRelations[number].forEach {
                    sum += counts[it][jump - 1]
                }
                counts[number][jump] = sum
            }
        }

        var result = BigInteger.valueOf(0)
        for (i in 0 until numberCount) {
            result += counts[i][n]
        }

        return (result.mod(BigInteger.valueOf(1000000007))).toInt()
    }
}


/////////////////////////////////////////////////////////////////////
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        try {
            log(Solution935().knightDialer(1))
            log(Solution935().knightDialer(2))
            log(Solution935().knightDialer(3))
            log(Solution935().knightDialer(4))
            log(Solution935().knightDialer(3131))
        } catch (e: Exception) {
            print(e)
        }
    }

    fun log(message: Any) {
        Log.i("lc", message.toString())
    }
}

/////////////////////////////////////////////////////////////////////