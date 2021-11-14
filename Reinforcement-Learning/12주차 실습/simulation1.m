% _Copyright 2019 The MathWorks, Inc._

%% Create MDP Environment
% Create an MDP model with eight states and two actions ("up" and "down").

MDP = createMDP(8,["up";"down"]);

%% 

MDP.T(1,2,1) = 1;
MDP.R(1,2,1) = 3;
MDP.T(1,3,2) = 1;
MDP.R(1,3,2) = 1;

% State 2 transition and reward
MDP.T(2,4,1) = 1;
MDP.R(2,4,1) = 2;
MDP.T(2,5,2) = 1;
MDP.R(2,5,2) = 1;
% State 3 transition and reward
MDP.T(3,5,1) = 1;
MDP.R(3,5,1) = 2;
MDP.T(3,6,2) = 1;
MDP.R(3,6,2) = 4;
% State 4 transition and reward
MDP.T(4,7,1) = 1;
MDP.R(4,7,1) = 3;
MDP.T(4,8,2) = 1;
MDP.R(4,8,2) = 2;
% State 5 transition and reward
MDP.T(5,7,1) = 1;
MDP.R(5,7,1) = 1;
MDP.T(5,8,2) = 1;
MDP.R(5,8,2) = 9;
% State 6 transition and reward
MDP.T(6,7,1) = 1;
MDP.R(6,7,1) = 5;
MDP.T(6,8,2) = 1;
MDP.R(6,8,2) = 1;
% State 7 transition and reward
MDP.T(7,7,1) = 1;
MDP.R(7,7,1) = 0;
MDP.T(7,7,2) = 1;
MDP.R(7,7,2) = 0;
% State 8 transition and reward
MDP.T(8,8,1) = 1;
MDP.R(8,8,1) = 0;
MDP.T(8,8,2) = 1;
MDP.R(8,8,2) = 0;
%% 
% Specify states |"s7"| and |"s8"| as terminal states of the MDP.

MDP.TerminalStates = ["s7";"s8"];
%% 
% Create the reinforcement learning MDP environment for this process model.

env = rlMDPEnv(MDP);
%% 
% To specify that the initial state of the agent is always state 1, specify 
% a reset function that returns the initial agent state. This function is called 
% at the start of each training episode and simulation. Create an anonymous function 
% handle that sets the initial state to 1.

env.ResetFcn = @() 1;
%% 
% Fix the random generator seed for reproducibility.

rng(0)
%% Create Q-Learning Agent
% To create a Q-learning agent, first create a Q table using the observation 
% and action specifications from the MDP environment. Set the learning rate of 
% the representation to |1|.

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
qTable = rlTable(obsInfo, actInfo);

tableRep = rlRepresentation(qTable);
tableRep.Options.LearnRate = 1;
%% 
% Next, create a Q-learning agent using this table representation, configuring 
% the epsilon-greedy exploration. For more information on creating Q-learning 
% agents, see <docid:rl_ref#mw_b6721f63-a5be-47ea-9325-78f7b379ae30 |rlQAgent|> 
% and <docid:rl_ref#mw_383e9fbf-e99d-4af3-b250-49c6c4ae6f3f |rlQAgentOptions|>.

agentOpts = rlQAgentOptions;
agentOpts.DiscountFactor = 1;
agentOpts.EpsilonGreedyExploration.Epsilon = 0.9;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.01;
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
trainOpts.MaxEpisodes = 500;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 10;
trainOpts.ScoreAveragingWindowLength = 30;
%% 
% Train the agent using the <docid:rl_ref#mw_c0ccd38c-bbe6-4609-a87d-52ebe4767852 
% |train|> function. This may take several minutes to complete. To save time while 
% running this example, load a pretrained agent by setting |doTraining| to |false|. 
% To train the agent yourself, set |doTraining| to |true|.

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('genericMDPQAgent.mat','qAgent');
end
%% 
% 
%% Validate Q-Learning Results
 

Data = sim(qAgent,env);
cumulativeReward = sum(Data.Reward)
%% 
 

QTable = getLearnableParameterValues(getCritic(qAgent))
QTable{1}


Data.Action.MDPActions.Data  % 최적의 경로 확인
Data.Reward.Data   % 최적 경로 액션에 따른 보상액

%% 
