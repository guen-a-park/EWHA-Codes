%% Train DQN Agent to Balance Cart-Pole System
% _Copyright 2019 The MathWorks, Inc._

%% Create Environment Interface
% Create a predefined environment interface for the pendulum.

env = rlPredefinedEnv("CartPole-Discrete")

%% Fix the random generator seed for reproducibility.

rng(0)
%% Create DQN agent
% A DQN agent approximates the long-term reward given observations and actions 
% using a critic value function representation. To create the critic, first create 
% a deep neural network with two inputs, the state and action, and one output. 


statePath = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24,'Name','CriticStateFC2')];
actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(24,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

%% 
% View the critic network configuration.

% figure
% plot(criticNetwork)
%% 
% Specify options for the critic representation using <docid:rl_ref#mw_45ccf57d-64f0-4822-8000-3f0f44f2572e 
% |rlRepresentationOptions|>.

criticOpts = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1);
%% 
% Create the critic representation using the specified neural network and options. 
% You must also specify the action and observation info for the critic, which 
% you obtain from the environment interface. For more information, see <docid:rl_ref#mw_453a7f45-3761-4387-9e1d-4c90ed8b5b57 
% |rlRepresentation|>.

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);
%% 
% To create the DQN agent, first specify the DQN agent options using <docid:rl_ref#mw_d66ae7b0-1964-4f66-a2ee-cfc08fc3657e 
% |rlDQNAgentOptions|>.

agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"periodic", ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256);
%% 
% Then, create the DQN agent using the specified critic representation and agent 
% options. For more information, see <docid:rl_ref#mw_167d0061-c095-446e-828f-816916a0f227 
% |rlDQNAgent|>.

agent = rlDQNAgent(critic,agentOpts);
%% Train Agent
% To train the agent, first specify the training options. For this example, 
% use the following options:
%% 
% * Run each training episode for at most 1000 episodes, with each episode lasting 
% at most 200 time steps.
% * Display the training progress in the Episode Manager dialog box (set the 
% |Plots| option) and disable the command line display (set the |Verbose| option).
% * Stop training when the agent receives an average cumulative reward greater 
% than 195 over 10 consecutive episodes. At this point, the agent can balance 
% the pendulum in the upright position.
%% 
% For more information, see <docid:rl_ref#mw_1f5122fe-cb3a-4c27-8c80-1ce46c013bf0 
% |rlTrainingOptions|>.

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 500, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480); 
%% 
% The cart-pole system can be visualized with using the |plot| function during training or simulation.

plot(env)
%% 

doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('MATLABCartpoleDQN.mat','agent');
end
%% 
% 
%% Simulate DQN Agent
% To validate the performance of the trained agent, simulate it within the cart-pole 
% environment.The agent can balance the cart-pole even when simulation time increases to 500.
plot(env)
env.State
simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)
%% 
