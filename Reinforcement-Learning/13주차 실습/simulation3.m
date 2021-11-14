% _Copyright 2019 The MathWorks, Inc._
%% Train DQN Agent to Swing Up and Balance Pendulum
% 
% 
mdl = 'rlSimplePendulumModel';
open_system(mdl)


%% Create Environment Interface
% Create a predefined environment interface for the pendulum.

env = rlPredefinedEnv('SimplePendulumModel-Discrete')

env.ResetFcn = @(in)setVariable(in,'theta0',pi,'Workspace',mdl);
%% 
% Specify the simulation time |Tf| and the agent sample time |Ts| in seconds

Ts = 0.05;
Tf = 20;
%% 
% Fix the random generator seed for reproducibility.

rng(0)
%% Create DQN agent

statePath = [
    imageInputLayer([3 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(48,'Name','CriticStateFC2')];
actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(48,'Name','CriticActionFC1','BiasLearnRateFactor',0)];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

%% 

criticOptions = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1);
%% 

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOptions);
%% 

agentOptions = rlDQNAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',3000,... 
    'UseDoubleDQN',false,...
    'DiscountFactor',0.9,...
    'MiniBatchSize',64);
%% 

agent = rlDQNAgent(critic,agentOptions);
%% Train Agent

trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',100,...
    'MaxStepsPerEpisode',500,...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',-1100,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',-1100);
%% 

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainingOptions);
else
    % Load pretrained agent for the example.
    load('SimulinkPendulumDQN.mat','agent');
end

%% Simulate DQN Agent

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);

sum(experience.Reward.Data)  %평균누적보상액

