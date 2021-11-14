% _Copyright 2019 The MathWorks, Inc._
%% Train Reinforcement Learning Agent in Basic Grid World
% This example shows how to solve a grid world environment using reinforcement 
% learning by training Q-learning and SARSA agents. For more information on these 
% agents, see <docid:rl_ug#mw_0264db4d-9787-425a-8f8a-1426e5639719 Q-Learning 
% Agents> and <docid:rl_ug#mw_b47b5155-a09e-4adf-81ed-7a5795fd2bd2 SARSA Agents>, 
% respectively.
% 
% This grid world environment has the following configuration and rules:
%% 
% # A 5-by-5 grid world bounded by borders, with 4 possible actions (North=1, 
% South=2, East=3, West=4).
% # The agent begins from cell [2,1] (second row, first column).
% # The agent receives reward +10 if it reaches the terminal state at cell [5,5] 
% (blue).
% # The environment contains a special jump from cell [2,4] to cell [4,4] with 
% +5 reward.
% # The agent is blocked by obstacles (black cells).
% # All other actions result in -1 reward.
%% 
% 
% 
% 
%% Create Grid World Environment
% Create the basic grid world environment.

env = rlPredefinedEnv("BasicGridWorld");
%% 
% To specify the initial state of the agent is always [2,1], specify a reset 
% function that returns the initial agent state. This function is called at the 
% start of each training episode and simulation. The states are numbered starting 
% at position [1,1] and counting down the column. Therefore, create an anonymous 
% function handle that sets the initial state to |2|.

env.ResetFcn = @() 2;
%% 
% Fix the random generator seed for reproducibility.

rng(0)
%% Create Q-Learning Agent
% To create a Q-learning agent, first create a Q table using the observation 
% and action specifications from the grid world environment. Set the learn rate 
% of the representation to 1.

qTable = rlTable(getObservationInfo(env),getActionInfo(env));
tableRep = rlRepresentation(qTable);
tableRep.Options.LearnRate = 1;
%% 
% Next, create a Q-learning agent using this table representation, configuring 
% the epsilon-greedy exploration. For more information on creating Q-learning 
% agents, see <docid:rl_ref#mw_b6721f63-a5be-47ea-9325-78f7b379ae30 |rlQAgent|> 
% and <docid:rl_ref#mw_383e9fbf-e99d-4af3-b250-49c6c4ae6f3f |rlQAgentOptions|>.

agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = .04;
qAgent = rlQAgent(tableRep,agentOpts);
%% Train Q-Learning Agent
% To train the agent, first specify the training options. For this example, 
% use the following options:
%% 
% * Train for at most 200 episodes, with each episode lasting at most 50 time 
% steps.
% * Stop training when the agent receives an average cumulative reward greater 
% than 10 over 30 consecutive episodes.
%% 
% For more information, see <docid:rl_ref#mw_1f5122fe-cb3a-4c27-8c80-1ce46c013bf0 
% |rlTrainingOptions|>.

trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 50;
trainOpts.MaxEpisodes= 200;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 11;
trainOpts.ScoreAveragingWindowLength = 30;
%% 
% Train the Q-Learning agent using the <docid:rl_ref#mw_c0ccd38c-bbe6-4609-a87d-52ebe4767852 
% |train|> function. This may take several minutes to complete. To save time while 
% running this example, load a pretrained agent by setting |doTraining| to |false|. 
% To train the agent yourself, set |doTraining| to |true|.

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end
%% 
% The *Episode Manager* window opens and displays the training progress.
% 
% 
%% Validate Q-Learning Results
% To validate the training results, simulate the agent in the training environment.
% 
% Before running the simulation, visualize the environment and configure the 
% visualization to maintain a trace of the agent states.

plot(env)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;
%% 
% Simulate the agent in the environment using the <docid:rl_ref#mw_e6296379-23b5-4819-a13b-210681e153bf 
% sim> function.

Data=sim(qAgent,env)
%% 
% The agent trace shows that the agent successfully found the jump from state 
% [2,4] to cell [4,4].
%%
QTable = getLearnableParameterValues(getCritic(qAgent))
QTable{1}


Data.Action.MDPActions.Data  % 최적의 경로 확인
Data.Reward.Data           % 최적 경로 액션에 따른 보상액
   