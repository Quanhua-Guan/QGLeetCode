//
//  LC669.swift
//  QGLeetCode-Swift
//
//  Created by 宇园 on 2022/9/10.
//

import Foundation

class LC669 {
    /// 669. 修剪二叉搜索树
    /**
     * Definition for a binary tree node.
     */
    public class TreeNode {
        public var val: Int
        public var left: TreeNode?
        public var right: TreeNode?
        public init() { self.val = 0; self.left = nil; self.right = nil; }
        public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
        public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
            self.val = val
            self.left = left
            self.right = right
        }
    }
    
    class Solution {
        func trimBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> TreeNode? {
            let prehead = TreeNode(Int.max, root, nil)
            
            func trim(_ node: TreeNode?, _ parent: TreeNode) {
                if node == nil { return }
                
                let node = node!
                if low <= node.val && node.val <= high {
                    trim(node.left, node)
                    trim(node.right, node)
                } else if node.val < low {
                    if node === parent.left {
                        parent.left = node.right
                    } else {
                        parent.right = node.right
                    }
                    
                    trim(node.right, parent)
                } else { // node.val > high
                    if node === parent.left {
                        parent.left = node.left
                    } else {
                        parent.right = node.left
                    }
                    
                    trim(node.left, parent)
                }
            }
            
            trim(root, prehead)
            return prehead.left
        }
    }
}
