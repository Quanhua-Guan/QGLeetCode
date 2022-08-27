//
//  LC793.swift
//  QGLeetCode-Swift
//
//  Created by 宇园 on 2022/8/28.
//

import Foundation

class LC793 {
    class Solution {
        func preimageSizeFZF(_ k: Int) -> Int {
            return minNumberOfKZeroTail(k + 1) - minNumberOfKZeroTail(k)
        }
        
        // x!结尾0个数为k个, 且x最小
        func minNumberOfKZeroTail(_ k: Int) -> Int {
            var l:Int64 = 0
            var r:Int64 = 5 * Int64(k)
            while l <= r {
                let m = (l + r) / 2
                if (zeroTailCount(m) < k) {
                    l = m + 1
                } else {
                    r = m - 1
                }
            }
            return Int(r + 1)
        }
        
        func zeroTailCount(_ x: Int64) -> Int64 {
            var n: Int64 = x
            var cnt: Int64 = 0
            while n > 0 {
                n /= 5
                cnt += n
            }
            return cnt
        }
    }
}
