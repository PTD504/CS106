# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, Queue
from game import Directions
import random, util

from game import Agent
from layout import Layout

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        def minimax_search(state, agentIdx, depth):
            legalActions = state.getLegalActions(agentIdx)

            if depth > self.depth or len(legalActions) == 0:
                return self.evaluationFunction(state)
            else:
                successors = [state.generateSuccessor(agentIdx, action) for action in legalActions]

            if agentIdx == 0:
                agentIdx = 1
                compare_function = max
            else:
                agentIdx = (agentIdx + 1) % state.getNumAgents()
                compare_function = min
                depth = depth + 1 if agentIdx == 0 else depth

            scores = [minimax_search(successor, agentIdx, depth) for successor in successors]

            return compare_function(scores)

        legalActions = gameState.getLegalActions()
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        bestScore = None
        bestActionIdx = []

        for i in range(len(successors)):
            score = minimax_search(successors[i], 1, 1)
            if bestScore is None or bestScore < score:
                bestScore = score
                bestActionIdx = [i]
            elif bestScore == score:
                bestActionIdx.append(i)

        return legalActions[random.choice(bestActionIdx)]

        '''def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        action = minimax(gameState)

        return action'''

        '''def minimax_search(state, agentIndex, depth):
            # if in min layer and last ghost
            if agentIndex == state.getNumAgents():
                # if reached max depth, evaluate state
                if depth == self.depth:
                    return self.evaluationFunction(state)
                # otherwise start new max layer with bigger depth
                else:
                    return minimax_search(state, 0, depth + 1)
            # if not min layer and last ghost
            else:
                moves = state.getLegalActions(agentIndex)
                # if nothing can be done, evaluate the state
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                # get all the minimax values for the next layer with each node being a possible state after a move
                next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

                # if max layer, return max of layer below
                if agentIndex == 0:
                    return max(next)
                # if min layer, return min of layer below
                else:
                    return min(next)
        # select the action with the greatest minimax value
        result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        return result'''


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    inf = 1000000007

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax_search_with_AB(state, agentIdx, depth, alpha, beta):
            if depth > self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIdx)
            successors = [state.generateSuccessor(agentIdx, action) for action in legalActions]
            nextAgentIdx = (agentIdx + 1) % state.getNumAgents()
            depth = depth + 1 if nextAgentIdx == 0 else depth

            if agentIdx == 0:
                maxScore = -self.inf

                for successor in successors:
                    score = minimax_search_with_AB(successor, nextAgentIdx, depth, alpha, beta)
                    maxScore = max(maxScore, score)
                    alpha = max(alpha, maxScore)

                    if beta <= alpha:
                        break

                return maxScore
            else:
                minScore = self.inf

                for successor in successors:
                    score = minimax_search_with_AB(successor, nextAgentIdx, depth, alpha, beta)
                    minScore = min(minScore, score)
                    beta = min(beta, minScore)

                    if beta <= alpha:
                        break

                return minScore


        legalActions = gameState.getLegalActions()
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]

        bestScore = None
        bestActionIdx = []

        for idx in range(len(successors)):
            score = minimax_search_with_AB(successors[idx], 1, 1, alpha=-self.inf, beta=self.inf)
            if bestScore is None or bestScore < score:
                bestScore = score
                bestActionIdx = [idx]
            elif bestScore == score:
                bestActionIdx.append(idx)

        return legalActions[random.choice(bestActionIdx)]

        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax_search(state, agentIdx, depth):
            legalActions = state.getLegalActions(agentIdx)

            if depth > self.depth or len(legalActions) == 0:
                return self.evaluationFunction(state)

            successors = [state.generateSuccessor(agentIdx, action) for action in legalActions]
            agentIdx = (agentIdx + 1) % state.getNumAgents()
            depth = depth + 1 if agentIdx == 0 else depth
            prob = 1 / len(legalActions)
            expectScore = 0

            for successor in successors:
                score = expectimax_search(successor, agentIdx, depth)
                if agentIdx == 0:
                    expectScore = max(expectScore, score)
                else:
                    expectScore += score * prob

            return expectScore

        legalActions = gameState.getLegalActions()
        successors = [gameState.generateSuccessor(0, action) for action in legalActions]
        scores = [expectimax_search(successor, 1, 1) for successor in successors]
        bestScore = max(scores)
        bestScores = [idx for idx in range(len(scores)) if scores[idx] == bestScore]

        #return legalActions[random.choice(bestScores)]
        return legalActions[random.choice(bestScores)]

        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if newCapsules:
        closestCapsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
    else:
        closestCapsule = 0

    if closestCapsule:
        closest_capsule = -3 / closestCapsule
    else:
        closest_capsule = 100

    if closestGhost:
        ghost_distance = -2 / closestGhost
    else:
        ghost_distance = -500

    foodList = newFood.asList()
    if foodList:
        closestFood = min([manhattanDistance(newPos, food) for food in foodList])
    else:
        closestFood = 0

    return -2 * closestFood + ghost_distance - 10 * len(foodList) + closest_capsule

def student_betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Nếu trạng thái hiện tại là trạng thái thua thì hàm đánh giá sẽ trả về số điểm đánh giá nhỏ nhất (thấp nhất ở đây là -Inf)
    if currentGameState.isLose():
        return float("-Inf")
    # Nếu trạng thái hiện tại là trạng thái chiến thắng thì hàm đánh giá sẽ trả về số điểm đánh giá cao nhất (cao nhất ở đây là Inf)
    if currentGameState.isWin():
        return float("Inf")

    # Một số thông tin cần lấy từ trạng thái hiện tại
    newPos = currentGameState.getPacmanPosition()
    newFood = set(currentGameState.getFood().asList())
    ScaredGhost = set() # Các con ma gây hại cho pacman
    nonScaredGhost = set() # Các con ma bị vô hiệu hóa, không gây hại
    newCapsules = set(currentGameState.getCapsules())

    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer == 0: # Những con ma gây hại là những con ma có scaredTimer = 0
            ScaredGhost.add(ghost.getPosition())
        else: # Những con ma không gây hại là những con ma có scaredTimer > 0
            nonScaredGhost.add(ghost.getPosition())

    # Lấy kích thước của bản đồ, phục vụ cho hàm tìm kiếm BFS
    row = Layout.getWidth()
    col = Layout.getHeight()
    # 
    check = [[0 for j in range(col + 1)] for i in range(row + 1)]
    queue = Queue()
    queue.push((newPos[0], newPos[1]))
    check[newPos[0]][newPos[1]] = 1
    # food_dis: là khoảng cách từ chấm thức ăn gần nhất đến pacman
    # ghost_dis: là khoảng cách của con ma gần nhất đến pacman
    # capsule_dis: là khoảng cách của chấm năng lượng gần nhất đến vị trí của pacman
    food_dis, ghost_dis, capsule_dis, nonScaredGhost_dis = -1, -1, -1, -1
    distance = 0
    dir = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    if len(ScaredGhost) == 0:
        ghost_dis = 5000
    if len(newCapsules) == 0:
        capsule_dis = float("Inf")
    if len(nonScaredGhost) == 0:
        nonScaredGhost_dis = 0

    # Thuật toán BFS tìm kiếm khoảng cách đường đi từ pacman đến các đối tượng khác, trong trường hợp này - mỗi bước đi có chi phí giống nhau - thuật toán BFS sẽ tìm được lời giải tối ưu nếu có (quãng đường tìm được sẽ là ngắn nhất)
    while queue.isEmpty() == False:
        size = queue.getSize()

        for _ in range(size):
            x, y = queue.pop()

            if food_dis < 0 and (x, y) in newFood:
                food_dis = distance
            if ghost_dis < 0 and (x, y) in ScaredGhost:
                ghost_dis = distance
            if capsule_dis < 0 and (x, y) in newCapsules:
                capsule_dis = distance
            if nonScaredGhost_dis < 0 and (x, y) in nonScaredGhost:
                nonScaredGhost_dis = distance

            if food_dis >= 0 and ghost_dis >= 0 and capsule_dis >= 0 and nonScaredGhost_dis >= 0:
                queue.clear()
                break

            for item in dir:
                u, v = x + item[0], y + item[1]

                if u > 0 and u <= row and v > 0 and v <= col and check[u][v] == 0 and currentGameState.hasWall(u, v) == False:
                    queue.push((u, v))
                    check[u][v] = 1

        distance += 1

    remain_food = len(newFood)
    remain_capsules = len(newCapsules)
    score = currentGameState.getScore()
    
    cons = max(row * col - len(newFood), 100)

    return 100 / food_dis - 10 / ghost_dis - cons * remain_food - 5 * remain_capsules + score + 2 / capsule_dis - 10 * nonScaredGhost_dis

# Abbreviation
better = student_betterEvaluationFunction
