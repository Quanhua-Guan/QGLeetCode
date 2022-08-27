import Foundation

class LC658 {
    class Solution {
        
        func bsle(_ nums: [Int], _ target: Int, _ from: Int = 0, _ to: Int? = nil) -> Int {
            var l = from
            var r = to ?? (nums.count - 1)
            while (l <= r) {
                let m = l + (r - l) / 2
                if nums[m] > target {
                    r = m - 1
                } else { // nums[m] <= target
                    if m == nums.count - 1 || nums[m + 1] > target {
                        return m
                    } else {
                        l = m + 1
                    }
                }
            }
            return -1
        }
        
        func findClosestElements(_ nums: [Int], _ k: Int, _ x: Int) -> [Int] {
            let le = bsle(nums, x)
            if le == -1 {
                return nums[0..<k].map {$0}
            } else if le == nums.count - 1 {
                return nums[nums.count - k..<nums.count].map{$0}
            } else {
                // 范围 [(l+1)..<r] 代表已选择的数, 所以已选数个数为 r-(l+1), 未选个数为 k-(r-(l+1))
                var l = le
                var r = le + 1
                while r - (l + 1) < k { // 已选个数小于 k
                    if r == nums.count {
                        l = nums.count - k - 1
                        continue
                    } else if l == -1 {
                        r = k
                        continue
                    }
                    
                    // m 代表未选个数的一半
                    // 本次迭代希望选中 m 个数加入已选数字
                    let m = max(1, (k - (r - (l + 1))) / 2)
                    let ll = max(0, l - (m - 1))
                    let rr = min(nums.count - 1, r + (m - 1))
                    
                    if x - nums[ll] <= nums[rr] - x {
                        l = ll - 1
                    } else {
                        r = rr + 1
                    }
                }
                return nums[(l + 1)..<r].map{$0}
            }
        }
    }
}

