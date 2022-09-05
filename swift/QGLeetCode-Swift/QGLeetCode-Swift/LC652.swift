//
//  LC652.swift
//  QGLeetCode-Swift
//
//  Created by 宇园 on 2022/9/5.
//

import Foundation


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

extension TreeNode : Hashable, Equatable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(val)
    }
    
    public static func == (lhs: TreeNode, rhs: TreeNode) -> Bool {
        return lhs === lhs
    }
}

class LC652 {
    
    class Solution {
        func findDuplicateSubtrees(_ root: TreeNode?) -> [TreeNode?] {
            var id = 0
            var seen: [String:(node: TreeNode, hash: Int)] = [:]
            var repeatSubtrees:Set<TreeNode> = []
            
            func dfs(_ node: TreeNode?) -> Int {
                if node == nil {
                    return 0
                }
                let key = "\(node!.val),\(dfs(node!.left)),\(dfs(node!.right))"
                if seen[key] != nil {
                    let pair = seen[key]!
                    repeatSubtrees.insert(pair.node)
                    return pair.hash
                } else {
                    id += 1
                    seen[key] = (node!, id)
                    return id
                }
            }
            
            let _ = dfs(root)
            return repeatSubtrees.map { $0 }
        }
    }
}
