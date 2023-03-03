%% Create Cart-Pole MATLAB Environment Interface

env = rlPredefinedEnv("CartPole-Discrete"); %SimplePendulumWithImage-Discrete /CartPole-Discrete

env.PenaltyForFalling = -15;
env.XThreshold = 2;
%% 
% Obtain the observation and action information from the environment interface. 

obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);

%% 
% Fix the random generator seed for reproducibility.

rng(0)
%% Create AC Agent

criticNetwork = [
    featureInputLayer(4,'Normalization','none','Name','state')
    fullyConnectedLayer(32,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(1, 'Name', 'CriticFC')];

criticOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);

critic = rlValueRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);
%% 

actorNetwork = [
    featureInputLayer(4,'Normalization','none','Name','state')
    fullyConnectedLayer(32, 'Name','ActorStateFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(2,'Name','action')];

actorOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);

actor = rlStochasticActorRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);
%% 

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32,...
    'EntropyLossWeight',0.01,...
    'DiscountFactor',0.99);
%% 

agent = rlACAgent(actor,critic,agentOpts);
%% Parallel Training Options

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode', 500,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',500,...
    'ScoreAveragingWindowLength',15); 
%% 
% You can visualize the cart-pole system can during training or simulation using 
% the |plot| function.

plot(env)
%% 

trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = "async";
trainOpts.ParallelizationOptions.DataToSendFromWorkers = "gradients";
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;


%% Train Agent

doTraining = true;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load('MATLABCartpoleParAC.mat','agent');
end

%% Simulate AC Agent
% You can visualize the cart-pole system with the plot function during simulation. 

plot(env)
%% 

simOptions = rlSimulationOptions('MaxSteps',1000);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)
