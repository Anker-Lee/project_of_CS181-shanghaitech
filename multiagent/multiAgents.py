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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent

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
        # Collect legal moves and child states
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

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        ghostPos = childGameState.getGhostPositions()
        # print('ghostPos:', ghostPos)
        if newPos in ghostPos: return -99999
        nearestFood = 99999
        capsulePos = childGameState.getCapsules()
        foodPos = list(newFood.asList())

        if foodPos != []:
            nearestFood = min( util.manhattanDistance( newPos, xy_f ) for xy_f in foodPos )

        return childGameState.getScore() + 1 / float(nearestFood)

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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        evlFunc = self.evaluationFunction
        currentDepth = 0
        legalMoves = gameState.getLegalActions(0)
        pacIndex = 0
        firstGhostIndex = 1

        def maxValue(gameState, currentDepth):
            values = [value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth) for action in gameState.getLegalActions(0)]
            return max(values)

        def minValue(gameState, agentIndex, currentDepth):
            if agentIndex < gameState.getNumAgents() - 1:
                values = [value(gameState.getNextState(agentIndex, action), agentIndex + 1, currentDepth) \
                        for action in gameState.getLegalActions(agentIndex)]
                return min(values)
            else: # agentIndex = numAgent - 1  i.e., last ghost
                values = [value(gameState.getNextState(agentIndex, action), pacIndex, currentDepth + 1) \
                        for action in gameState.getLegalActions(agentIndex)] # 当下一个 agent 又变为 Pac-Man 时证明当前深度结束
                return min(values)

        def value(gameState, agentIndex, currentDepth):
            if gameState.isWin() or gameState.isLose() or currentDepth == depth:
                return evlFunc(gameState)
            elif agentIndex == pacIndex:
                return maxValue(gameState, currentDepth)
            else: 
                return minValue(gameState, agentIndex, currentDepth)

        values = [value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth) for action in legalMoves]
        maxV = max(values)
        maxIndices = [i for i in range(len(values)) if values[i] == maxV]
        chosenIndex = random.choice(maxIndices)
        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        evlFunc = self.evaluationFunction
        currentDepth = 0
        legalMoves = gameState.getLegalActions(0)
        pacIndex = 0
        firstGhostIndex = 1

        alpha = -99999
        beta = 99999
        def maxValue(gameState, currentDepth, alpha, beta):
            maxV = -99999
            for action in gameState.getLegalActions(0):
                maxV = max( maxV, value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth, alpha, beta) )
                if maxV > beta: break
                alpha = max( alpha, maxV )
            return maxV

        def minValue(gameState, agentIndex, currentDepth, alpha, beta):
            minV = 99999
            if agentIndex < gameState.getNumAgents() - 1: # agent index < num of ghost
                for action in gameState.getLegalActions(agentIndex):
                    minV = min( minV, value(gameState.getNextState(agentIndex, action), agentIndex + 1, currentDepth, alpha, beta) )
                    if minV < alpha: break
                    beta = min( beta, minV )
                return minV
            else: # agentIndex = numAgent - 1  i.e., last ghost
                for action in gameState.getLegalActions(agentIndex):
                    minV = min( minV, value(gameState.getNextState(agentIndex, action), pacIndex, currentDepth + 1, alpha, beta) )
                    if minV < alpha: break
                    beta = min( beta, minV )
                return minV

        def value(gameState, agentIndex, currentDepth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or currentDepth == depth:
                return evlFunc(gameState)
            elif agentIndex == pacIndex:
                return maxValue(gameState, currentDepth, alpha, beta)
            else: # agent is ghost
                return minValue(gameState, agentIndex, currentDepth, alpha, beta)

        maxV = -99999
        actionMap = {} # value map to action
        for action in legalMoves:
            tmpV = value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth, max( alpha, maxV ), beta)
            actionMap[tmpV] = action
            maxV = max( maxV, tmpV )
        return actionMap[maxV]

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
        depth = self.depth
        evlFunc = self.evaluationFunction
        currentDepth = 0
        legalMoves = gameState.getLegalActions(0)
        pacIndex = 0
        firstGhostIndex = 1

        def maxValue(gameState, currentDepth):
            values = [value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth) for action in gameState.getLegalActions(0)]
            return max(values)

        def expectedValue(gameState, agentIndex, currentDepth):
            if agentIndex < gameState.getNumAgents() - 1:
                values = [value(gameState.getNextState(agentIndex, action), agentIndex + 1, currentDepth) \
                        for action in gameState.getLegalActions(agentIndex)]
                return float(sum(values)) / float(len(values)) # expectation is the average of values
            else: # agentIndex = numAgent - 1  i.e., last ghost
                values = [value(gameState.getNextState(agentIndex, action), pacIndex, currentDepth + 1) \
                        for action in gameState.getLegalActions(agentIndex)] # 当下一个 agent 又变为 Pac-Man 时证明当前深度结束
                return float(sum(values)) / float(len(values)) # expectation is the average of values

        def value(gameState, agentIndex, currentDepth):
            if gameState.isWin() or gameState.isLose() or currentDepth == depth:
                return evlFunc(gameState)
            elif agentIndex == pacIndex:
                return maxValue(gameState, currentDepth)
            else: 
                return expectedValue(gameState, agentIndex, currentDepth)

        values = [value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth) for action in legalMoves]
        maxV = max(values)
        maxIndices = [i for i in range(len(values)) if values[i] == maxV]
        chosenIndex = random.choice(maxIndices)
        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: < In q1, we designed the evaluation function by the score of next legal action 
    and the distance from the Pac-man to the closest food. We take the sum of the score and the reciprocal of
    the distance as the final score of next legal action. Based on the evaluation function in q1, we consider a new 
    metric, i.e, scared times. We take the sum of each scared time for each ghost as a new metric.
    The higher the new metric is, the better the state is.>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    "*** YOUR CODE HERE ***"
    ghostPos = currentGameState.getGhostPositions()
    # print('ghostPos:', ghostPos)
    if pacPos in ghostPos: return -99999
    nearestFood = 99999
    foodPos = list(food.asList())
    if foodPos != []:
        nearestFood = min( util.manhattanDistance( pacPos, xy_f ) for xy_f in foodPos )

    return currentGameState.getScore() + 1 / float(nearestFood) + sum(scaredTimes)

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def __init__(self):
        self.parameters = [ [ 2, 40,  9,  8, 60,  3, 80],\
                            [40, 60, 20, 20, 60, 6, 30],\
                            [50, 70, 700, 600, 700, 7, 100],\
                            [  3,   3,   8, 700,  20,  30,  90],\
                            [ 40,  50,  30,   1, 500,   3,   9],\
                            [  1,   8,  1,  30,  10, 100, 600],\
                            [ 80,  70, 700,  30,   9,   2,   1],\
                            [  3,  50,  80, 200,  30,  90,  20],\
                            [100,  70, 700,  40, 500,   2,   7],\
                            [  9,   9,  70,  30, 200,   4,  40]]
        self.roundNum = -1
        self.p = [0,0,0,0,0,0,0]

    def myEvaluationFunction(self, currentGameState):
        pacPos = currentGameState.getPacmanPosition()
        food = currentGameState.getFood()
        foodPos = list(food.asList())
        capsulePos = currentGameState.getCapsules()
        ghostStates = currentGameState.getGhostStates()
        ghostPos = currentGameState.getGhostPositions()
        
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

        if currentGameState.isLose(): 
            return -99999
        
        d2f = [util.manhattanDistance(pacPos, xy_f) for xy_f in foodPos]

        if len(d2f) != 0:
            s_closestF = 1 / min(d2f)       # the smaller, the better
            s_numFood = 1 / len(d2f)        # the smaller, the better
        else:
            s_closestF = 0
            s_numFood = 0

        d2cap = [util.manhattanDistance(pacPos, xy_c) for xy_c in capsulePos]
        if len(d2cap) == 0: 
            s_numCap = 0
            s_closestCap = 0
        else:
            s_closestCap = 1 / min(d2cap)   # the smaller, the better
            s_numCap = 1 / len(d2cap)       # the smaller, the better

        d2ghost = [util.manhattanDistance(pacPos, xy_g) for xy_g in ghostPos]
        d2dangerG = []
        d2safeG = []

        s_numSafeG = 0
        for i, scaredT in enumerate(scaredTimes):
            if d2ghost[i] < scaredT / 2: # able to catch the ghost
                s_numSafeG += 1
                d2safeG.append(d2ghost[i])
            else:
                d2dangerG.append(d2ghost[i])

        s_d2dangerG = 0
        if len(d2dangerG) != 0: 
            s_d2dangerG = min(d2dangerG)

        s_stateScore = currentGameState.getScore()
        return self.p[0] * s_stateScore + self.p[1] * s_closestF + self.p[6] * s_numFood + self.p[2] * s_closestCap \
                + self.p[3]* s_numCap + self.p[4] * s_numSafeG - self.p[5] * s_d2dangerG

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """ 
        "*** YOUR CODE HERE ***"
        depth = 3
        evlFunc = self.myEvaluationFunction
        currentDepth = 0
        legalMoves = gameState.getLegalActions(0)
        pacIndex = 0
        firstGhostIndex = 1
        # paraLib = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]
        paraLib = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700]
        # paraLib = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90]

        if gameState.getScore() == 0 and len(list(gameState.getFood().asList())) == 69:
            self.roundNum += 1
        
        if self.roundNum == 0:
            # self.p = self.parameters[self.roundNum][:]
            # self.parameters[self.roundNum] = np.random.choice(paraLib, size=7)
            self.p = [ 2, 40,  9,  8, 60,  3, 80]
        elif self.roundNum == 1:
            self.p = [40, 60, 20, 20, 60, 6, 30]
        elif self.roundNum == 2:
            self.p = [ 50,  70, 700, 600, 700,   7, 100]
        elif self.roundNum == 3:
            self.p = [  3,   3,   8, 700,  20,  30,  90]
        elif self.roundNum == 4:
            self.p = [ 40,  50,  30,   1, 500,   3,   9]
        elif self.roundNum == 5:
            self.p = [  1,   8,  1,  30,  10, 100, 600]
        elif self.roundNum == 6:
            self.p = [ 80,  70, 700,  30,   9,   2,   1]
        elif self.roundNum == 7:
            self.p = [  3,  50,  80, 200,  30,  90,  20]
        elif self.roundNum == 8:
            self.p = [100,  70, 700,  40, 500,   2,   7]
        elif self.roundNum == 9:
            self.p = [  9,   9,  70,  30, 200,   4,  40]


        def maxValue(gameState, currentDepth):
            values = [value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth) for action in gameState.getLegalActions(0)]
            return max(values)

        def minValue(gameState, agentIndex, currentDepth):
            if agentIndex < gameState.getNumAgents() - 1:
                values = [value(gameState.getNextState(agentIndex, action), agentIndex + 1, currentDepth) \
                        for action in gameState.getLegalActions(agentIndex)]
                return min(values)
            else: # agentIndex = numAgent - 1  i.e., last ghost
                values = [value(gameState.getNextState(agentIndex, action), pacIndex, currentDepth + 1) \
                        for action in gameState.getLegalActions(agentIndex)] # 当下一个 agent 又变为 Pac-Man 时证明当前深度结束
                return min(values)

        def value(gameState, agentIndex, currentDepth):
            if gameState.isWin() or gameState.isLose() or currentDepth == depth:
                return evlFunc(gameState)
            elif agentIndex == pacIndex:
                return maxValue(gameState, currentDepth)
            else: 
                return minValue(gameState, agentIndex, currentDepth)

        values = [value(gameState.getNextState(pacIndex, action), firstGhostIndex, currentDepth) for action in legalMoves]
        maxV = max(values)
        maxIndices = [i for i in range(len(values)) if values[i] == maxV]
        chosenIndex = random.choice(maxIndices)
        return legalMoves[chosenIndex]