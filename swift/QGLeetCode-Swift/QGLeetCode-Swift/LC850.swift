//
//  LC850.swift
//  QGLeetCode-Swift
//
//  Created by 宇园 on 2022/9/16.
//

import Foundation

class LC850 {
    /// 850. 矩形面积 II
    class Solution {
        func rectangleArea(_ rectangles: [[Int]]) -> Int {
            let MOD = 1000_000_007
            
            var list: [Int] = []
            for rect in rectangles {
                list.append(rect[0])
                list.append(rect[2])
            }
            list.sort()

            var ans = 0
            for i in 1..<list.count {
                let a = list[i - 1]
                let b = list[i]
                let len = b - a
                if len == 0 {
                    continue
                }

                var lines: [[Int]] = []
                for rect in rectangles {
                    if rect[0] <= a && b <= rect[2] {
                        lines.append([rect[1], rect[3]])
                    }
                }
                lines.sort { l1, l2 in
                    l1[0] != l2[0] ? l1[0] < l2[0] : l1[1] < l2[1]
                }
                
                var total = 0
                var l = -1
                var r = -1
                for line in lines {
                    if line[0] > r {
                        total += r - l
                        l = line[0]
                        r = line[1]
                    } else if line[1] > r {
                        r = line[1]
                    }
                }
                total += r - l
                ans += total * len
                ans %= MOD
            }
            
            return ans
        }
    }
}
