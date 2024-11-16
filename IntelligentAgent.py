from BaseAI import BaseAI
from Grid import Grid
import math

class IntelligentAgent(BaseAI):
    def __init__(self):
        self.depth = 4
        # 初始化启发式函数的权重
        self.weights = {
            'snake_score': 1.5,
            'monotonicity_score': 2.0,
            'empty_tiles': 2.5,
            'merges': 2,
            'smoothness_score': 2.0,
        }

    def evaluate(self, grid):
        # 使用提供的启发式函数计算评估分数
        weight_matrix = [[4**16, 4**14, 4**14, 4**13],
                         [4**9, 4**10, 4**11, 4**12],
                         [4**8, 4**7, 4**6, 4**5],
                         [4**1, 4**2, 4**3, 4**4]]
        '''weight_matrix = [[16,14, 13, 12]
                        ,[8, 9, 10, 11]
                        ,[7, 6, 5, 4]
                        ,[0, 1, 2, 3]]'''
        ''' weight_matrix = [[2**16, 2**14, 2**13, 2**12],
                         [2**8, 2**9, 2**10, 2**11],
                            [2**7, 2**6, 2**5, 2**4],
                            [2**0, 2**1, 2**2, 2**3]]'''
                         
        snake_score = 0 
        for i in range(4):  
            for j in range(4):  
                snake_score += grid.map[i][j] * weight_matrix[i][j]  # 累加每个元素与其权重的乘积到 snake_score
        monotonicity_score = self.calculate_monotonicity(grid)
        empty_tiles = len(grid.getAvailableCells())
        merges = self.count_merges(grid)
        smoothness_score = self.calculate_smoothness(grid)
        
        # 计算最终的启发式分数
        score = (self.weights['snake_score'] * snake_score +
                 self.weights['monotonicity_score'] * monotonicity_score +
                 self.weights['empty_tiles'] * empty_tiles +
                 self.weights['merges'] * merges +
                 self.weights['smoothness_score'] * smoothness_score)
        
        return score

    def calculate_monotonicity(self, grid):
        # 计算单调性分数
        monotonicity_score = 0
        
        # 遍历行和列，分别计算单调性
        for i in range(4):
            row_score = sum(
                (grid.map[i][j-1] - grid.map[i][j]) 
                if grid.map[i][j-1] > grid.map[i][j] 
                else (grid.map[i][j] - grid.map[i][j-1]) * -1
                for j in range(1, 4)
            )
            
            col_score = sum(
                (grid.map[j-1][i] - grid.map[j][i]) 
                if grid.map[j-1][i] > grid.map[j][i] 
                else (grid.map[j][i] - grid.map[j-1][i]) * -1
                for j in range(1, 4)
            )
            
            monotonicity_score += row_score + col_score
        
        return monotonicity_score


    def count_merges(self, grid):
        # 计算可能的合并次数
        merges = 0
        for x in range(4):
            for y in range(4):
                if x < 3 and grid.map[x][y] == grid.map[x+1][y]:
                    merges += 1
                if y < 3 and grid.map[x][y] == grid.map[x][y+1]:
                    merges += 1
        return merges

    def calculate_smoothness(self, grid):
        # 计算平滑性分数
        smoothness_score = 0
        for x in range(4):
            for y in range(4):
                if x < 3: 
                    smoothness_score -= abs(grid.map[x][y] - grid.map[x+1][y])
                if y < 3: 
                    smoothness_score -= abs(grid.map[x][y] - grid.map[x][y+1])
        return smoothness_score
    

    def expectiminimax(self, grid, depth, alpha, beta, isPlayerTurn):
        # 实现期望极小极大算法
        if depth == 0 or not grid.canMove():
            return None, self.evaluate(grid)
        
        if isPlayerTurn:  # 玩家回合（最大化得分）
            maxEval = float('-inf')
            bestMove = None
            # getAvailableMoves()返回一个列表，包含所有可能的移动方向和对应移动后grid
            for move, nextGrid in grid.getAvailableMoves():
                _, eval = self.expectiminimax(nextGrid, depth - 1, alpha, beta, False)
                if eval > maxEval:
                    maxEval = eval
                    bestMove = move
                alpha = max(alpha, eval)
                if beta <= alpha:  # alpha-beta 剪枝
                    break
            return bestMove, maxEval

        else:  # 随机回合（计算期望值 min）
            minEval = float('inf')
            #expectedValues = []
            blankCells = grid.getAvailableCells()
            
            # 计算每个空格的期望值
            for cell in blankCells:
                expectedValue = 0
                for value, prob in [(2, 0.9), (4, 0.1)]:  # 可能的值和对应概率
                    child = grid.clone()
                    child.insertTile(cell, value)
                    _, eval = self.expectiminimax(child, depth - 1, alpha, beta, True)
                    expectedValue += prob * eval
                #expectedValues.append(expectedValue)
                
                # 更新最小值
                minEval = min(minEval, expectedValue)
                beta = min(beta, expectedValue)
                if beta <= alpha:  # alpha-beta 剪枝
                    break

            return None, minEval
    
    def getMove(self, grid):
        # 实现 AI的决策函数，返回移动方向
        move, _ = self.expectiminimax(grid, self.depth, float('-inf'), float('inf'), True)
        return move

'''
    def expectiminimax(self, grid, depth, alpha, beta, playerSide):
        # 实现期望极小极大算法
        if depth == 0 or not grid.canMove():
            return None, self.evaluate(grid)
        
        if playerSide: #max  越大越好
            maxEval = float('-inf')
            bestMove = None
            # getAvailableMoves()返回一个列表，包含所有可能的移动方向和对应移动后grid
            for move, nextGrid in grid.getAvailableMoves():
                _, eval = self.expectiminimax(nextGrid, depth-1, alpha, beta, False)
                if eval > maxEval:
                    maxEval = eval
                    bestMove = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return bestMove, maxEval
        
        else:   #min  越小越好
            minEval = float('inf')
            blank_grids = grid.getAvailableMoves()
            # 随机从blank_grids中取一个cell，然后在这个cell上放置2或4，计算期望值
            for cell in blank_grids:
                expectedValue = 0
                child_2 = grid.clone()
                child_2.insertTile(cell, 2)
                _, eval_2 = self.expectiminimax(child_2, depth-1, alpha, beta, True)
                expectedValue += 0.9 * eval_2
                child_4 = grid.clone()
                child_4.insertTile(cell, 4)
                _, eval_4 = self.expectiminimax(child_4, depth-1, alpha, beta, True)
                # expectedValue把这个cell的是2和4的情况的期望值加起来，得到了这个cell的总期望值
                expectedValue += 0.1 * eval_4
                # 比较之前最小的期望值和这个cell的总期望值，取最小的
                minEval = min(expectedValue, minEval)
                beta = min(beta, expectedValue)
                if beta <= alpha:
                    break
            return None, minEval

'''